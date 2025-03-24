import csv
import time
from kubernetes import client, config

# Load Kubernetes configuration
config.load_kube_config()
v1 = client.CoreV1Api()

# Function to fetch logs for all pods
def get_pod_logs(namespace, pod_name):
    try:
        return v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)[:1000]  # Limit log size
    except Exception as e:
        return f"Error fetching logs: {e}"

# Function to collect pod status and logs
def get_pod_status():
    pods = v1.list_pod_for_all_namespaces(watch=False)
    data = []

    for pod in pods.items:
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace
        status = pod.status.phase
        restart_count = sum([c.restart_count for c in pod.status.container_statuses or []])
        reason = pod.status.reason if pod.status.reason else "N/A"

        # Detect failures (Readiness Probe, ImagePullBackOff, CrashLoopBackOff)
        conditions = pod.status.conditions or []
        failure_reasons = []
        for condition in conditions:
            if condition.type in ["Ready", "ContainersReady"] and condition.status == "False":
                failure_reasons.append(condition.reason)

        for container_status in pod.status.container_statuses or []:
            if container_status.state.waiting:
                failure_reasons.append(container_status.state.waiting.reason)

        failure_reasons = ", ".join(set(failure_reasons)) or "N/A"

        # Fetch logs for all pods (not just failing ones)
        pod_logs = get_pod_logs(namespace, pod_name)

        print(f"Namespace: {namespace}, Pod: {pod_name}, Status: {status}, Restarts: {restart_count}, Reason: {failure_reasons}, Logs: {pod_logs[:100]}...")  # Truncated logs for readability

        data.append([namespace, pod_name, status, restart_count, failure_reasons, pod_logs])

    return data

# Function to write data to CSV
def write_to_csv():
    filename = "pod_status_with_logs.csv"

    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)

        if file.tell() == 0:
            writer.writerow(["Namespace", "Pod Name", "Status", "Restart Count", "Failure Reasons", "Logs"])

        while True:
            pod_data = get_pod_status()
            if pod_data:
                writer.writerows(pod_data)
                file.flush()  # Ensure data is written immediately
                print(f"✅ Data successfully written to {filename}")
            else:
                print("⚠️ No pod data retrieved. Check cluster status.")

            time.sleep(60)  # Collect data every 60 seconds

# Run the data collection
write_to_csv()