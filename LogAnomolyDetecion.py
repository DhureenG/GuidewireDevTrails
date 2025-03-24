#!/usr/bin/env python3
"""
Docker Image Anomaly Detector

This script analyzes Docker images for potential security issues, configuration
problems, and statistical anomalies that might indicate compromised or 
problematic container images.
"""

import json
import subprocess
import os
import re
import sys
import statistics
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Tuple, Any, Optional, Set

class DockerImageAnalyzer:
    def __init__(self, image_name: str = None, all_images: bool = False, 
                 verbose: bool = False, threshold_multiplier: float = 2.0):
        """Initialize the Docker image analyzer.
        
        Args:
            image_name: Specific image to analyze
            all_images: Whether to analyze all images
            verbose: Enable detailed output
            threshold_multiplier: Multiplier for statistical anomaly thresholds
        """
        self.image_name = image_name
        self.all_images = all_images
        self.verbose = verbose
        self.threshold_multiplier = threshold_multiplier
        self.images = []
        self.anomalies = []
        
        # Common security issues to check
        self.security_patterns = {
            'root_user': r'USER\s+root',
            'suspicious_env': r'ENV\s+.*(PASSWORD|TOKEN|KEY|SECRET).*=',
            'suspicious_curl': r'RUN\s+.*curl.*\|\s*bash',
            'suspicious_wget': r'RUN\s+.*wget.*\|\s*bash',
            'suspicious_port': r'EXPOSE\s+(22|3389)'
        }
        
        # Known cryptominers and malicious packages
        self.suspicious_packages = [
            'xmrig', 'cgminer', 'cpuminer', 'coinhive', 'cryptonight',
            'stratum+tcp', 'monero', '.xmr', 'minerd', 'minergate'
        ]

    def run(self) -> None:
        """Main execution function."""
        if not self._check_docker_available():
            print("Error: Docker is not available or not running")
            sys.exit(1)
            
        if self.all_images:
            self._get_all_images()
        elif self.image_name:
            if not self._check_image_exists(self.image_name):
                print(f"Error: Image '{self.image_name}' not found")
                sys.exit(1)
            self.images = [self.image_name]
        else:
            print("Error: Please specify an image or use --all flag")
            sys.exit(1)
            
        if not self.images:
            print("No images found to analyze")
            sys.exit(0)
            
        # Get baseline statistics
        self._analyze_images()
        
        # Display results
        self._display_results()
    
    def _check_docker_available(self) -> bool:
        """Check if Docker is available on the system."""
        try:
            subprocess.run(['docker', 'info'], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, 
                           check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _check_image_exists(self, image_name: str) -> bool:
        """Check if a specific image exists."""
        try:
            result = subprocess.run(
                ['docker', 'image', 'inspect', image_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            return result.returncode == 0
        except subprocess.SubprocessError:
            return False

    def _get_all_images(self) -> None:
        """Get all Docker images from the local system."""
        try:
            result = subprocess.run(
                ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            self.images = [
                img for img in result.stdout.strip().split('\n') 
                if img and '<none>:<none>' not in img
            ]
            
            if self.verbose:
                print(f"Found {len(self.images)} images to analyze")
        except subprocess.SubprocessError as e:
            print(f"Error getting docker images: {e}")
            sys.exit(1)

    def _get_image_details(self, image_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific image."""
        try:
            result = subprocess.run(
                ['docker', 'image', 'inspect', image_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            details = json.loads(result.stdout)
            if details and isinstance(details, list):
                return details[0]
            return {}
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            print(f"Error inspecting image {image_name}: {e}")
            return {}

    def _get_image_history(self, image_name: str) -> List[Dict[str, Any]]:
        """Get the build history of an image."""
        try:
            result = subprocess.run(
                ['docker', 'image', 'history', '--no-trunc', '--format', '{{json .}}', image_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            history = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            return history
        except subprocess.SubprocessError as e:
            print(f"Error getting history for {image_name}: {e}")
            return []

    def _analyze_images(self) -> None:
        """Analyze all selected images for anomalies."""
        image_sizes = []
        layer_counts = []
        creation_dates = []
        
        # First pass: collect statistics
        for image in self.images:
            details = self._get_image_details(image)
            if not details:
                continue
                
            # Get size statistics
            if 'Size' in details:
                image_sizes.append(details['Size'])
                
            # Get layer statistics
            if 'RootFS' in details and 'Layers' in details['RootFS']:
                layer_counts.append(len(details['RootFS']['Layers']))
                
            # Get creation date
            if 'Created' in details:
                try:
                    creation_date = datetime.strptime(
                        details['Created'].split('.')[0], 
                        '%Y-%m-%dT%H:%M:%S'
                    )
                    creation_dates.append(creation_date)
                except (ValueError, TypeError):
                    pass
        
        # Calculate statistics
        if image_sizes:
            mean_size = statistics.mean(image_sizes)
            stdev_size = statistics.stdev(image_sizes) if len(image_sizes) > 1 else mean_size / 2
            size_threshold = mean_size + (stdev_size * self.threshold_multiplier)
        else:
            size_threshold = 0
            
        if layer_counts:
            mean_layers = statistics.mean(layer_counts)
            stdev_layers = statistics.stdev(layer_counts) if len(layer_counts) > 1 else mean_layers / 2
            layer_threshold = mean_layers + (stdev_layers * self.threshold_multiplier)
        else:
            layer_threshold = 0
        
        # Second pass: detect anomalies
        for image in self.images:
            image_anomalies = []
            
            details = self._get_image_details(image)
            if not details:
                continue
                
            # Check image size anomalies
            if 'Size' in details and size_threshold > 0:
                if details['Size'] > size_threshold:
                    image_anomalies.append({
                        'type': 'size',
                        'description': f"Unusually large image size: {self._format_size(details['Size'])}",
                        'severity': 'medium'
                    })
            
            # Check layer count anomalies
            if 'RootFS' in details and 'Layers' in details['RootFS'] and layer_threshold > 0:
                if len(details['RootFS']['Layers']) > layer_threshold:
                    image_anomalies.append({
                        'type': 'layers',
                        'description': f"Unusually high layer count: {len(details['RootFS']['Layers'])}",
                        'severity': 'low'
                    })
            
            # Check for recent creation (potentially suspicious)
            if 'Created' in details and creation_dates:
                try:
                    creation_date = datetime.strptime(
                        details['Created'].split('.')[0], 
                        '%Y-%m-%dT%H:%M:%S'
                    )
                    if (datetime.now() - creation_date) < timedelta(days=1):
                        image_anomalies.append({
                            'type': 'recent_creation',
                            'description': f"Recently created: {creation_date.isoformat()}",
                            'severity': 'info'
                        })
                except (ValueError, TypeError):
                    pass
            
            # Check Dockerfile instructions for security issues
            history = self._get_image_history(image)
            for layer in history:
                if 'CreatedBy' in layer:
                    command = layer['CreatedBy']
                    
                    # Check for security patterns
                    for pattern_name, pattern in self.security_patterns.items():
                        if re.search(pattern, command, re.IGNORECASE):
                            image_anomalies.append({
                                'type': 'security_pattern',
                                'description': f"Potential security issue: {pattern_name}",
                                'command': command,
                                'severity': 'high'
                            })
                    
                    # Check for suspicious packages
                    for package in self.suspicious_packages:
                        if package in command.lower():
                            image_anomalies.append({
                                'type': 'suspicious_package',
                                'description': f"Suspicious package detected: {package}",
                                'command': command,
                                'severity': 'critical'
                            })
            
            # Check for missing HEALTHCHECK
            has_healthcheck = False
            if 'ContainerConfig' in details and 'Healthcheck' in details['ContainerConfig']:
                has_healthcheck = True
            elif 'Config' in details and 'Healthcheck' in details['Config']:
                has_healthcheck = True
                
            if not has_healthcheck:
                image_anomalies.append({
                    'type': 'missing_healthcheck',
                    'description': "No HEALTHCHECK instruction found",
                    'severity': 'low'
                })
            
            # Check for exposed ports
            exposed_ports = set()
            if 'ContainerConfig' in details and 'ExposedPorts' in details['ContainerConfig']:
                exposed_ports.update(details['ContainerConfig']['ExposedPorts'].keys())
            elif 'Config' in details and 'ExposedPorts' in details['Config']:
                exposed_ports.update(details['Config']['ExposedPorts'].keys())
                
            if exposed_ports:
                suspicious_ports = {'22/tcp', '3389/tcp'}
                suspicious_found = suspicious_ports.intersection(exposed_ports)
                if suspicious_found:
                    image_anomalies.append({
                        'type': 'suspicious_ports',
                        'description': f"Suspicious ports exposed: {', '.join(suspicious_found)}",
                        'severity': 'high'
                    })
            
            if image_anomalies:
                self.anomalies.append({
                    'image': image,
                    'anomalies': image_anomalies
                })

    def _display_results(self) -> None:
        """Display the analysis results."""
        if not self.anomalies:
            print("\n✅ No anomalies detected in the analyzed images.")
            return
            
        print(f"\n⚠️  Detected anomalies in {len(self.anomalies)} of {len(self.images)} images:\n")
        
        for entry in self.anomalies:
            print(f"Image: {entry['image']}")
            
            for anomaly in entry['anomalies']:
                severity = anomaly['severity']
                severity_marker = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🔵',
                    'info': '⚪'
                }.get(severity, '•')
                
                print(f"  {severity_marker} [{severity.upper()}] {anomaly['description']}")
                
                if self.verbose and 'command' in anomaly:
                    print(f"    Command: {anomaly['command']}")
                    
            print()

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0 or unit == 'TB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0


def main():
    parser = argparse.ArgumentParser(description="Docker Image Anomaly Detector")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--image", help="Analyze a specific image")
    group.add_argument("-a", "--all", action="store_true", help="Analyze all images")
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-t", "--threshold", type=float, default=2.0,
                        help="Threshold multiplier for statistical anomalies (default: 2.0)")
    
    args = parser.parse_args()
    
    analyzer = DockerImageAnalyzer(
        image_name=args.image,
        all_images=args.all,
        verbose=args.verbose,
        threshold_multiplier=args.threshold
    )
    
    analyzer.run()


if __name__ == "__main__":
    main()