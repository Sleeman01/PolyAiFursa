import boto3
import json
import os
import paramiko
import io

REGION = os.environ["AWS_REGION"]
SECRET_NAME = os.environ["SSH_SECRET_NAME"]
CP_PUBLIC_IP = os.environ["CONTROL_PLANE_PUBLIC_IP"]

asg_client = boto3.client("autoscaling", region_name=REGION)
secrets_client = boto3.client("secretsmanager", region_name=REGION)
ec2_client = boto3.client("ec2", region_name=REGION)


def get_ssh_key():
    resp = secrets_client.get_secret_value(SecretId=SECRET_NAME)
    return paramiko.RSAKey.from_private_key(io.StringIO(resp["SecretString"]))


def get_node_name(instance_id):
    # kubeadm nodes are named after the private DNS hostname
    resp = ec2_client.describe_instances(InstanceIds=[instance_id])
    return resp["Reservations"][0]["Instances"][0]["PrivateDnsName"]


def delete_node(node_name):
    key = get_ssh_key()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(CP_PUBLIC_IP, username="ubuntu", pkey=key, timeout=15)
    cmd = f"sudo kubectl delete node {node_name} --ignore-not-found"
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_status = stdout.channel.recv_exit_status()
    ssh.close()
    return exit_status


def handler(event, context):
    for record in event["Records"]:
        message = json.loads(record["Sns"]["Message"])
        instance_id = message["EC2InstanceId"]
        lifecycle_hook_name = message["LifecycleHookName"]
        asg_name = message["AutoScalingGroupName"]

        try:
            node_name = get_node_name(instance_id)
            delete_node(node_name)
            result = "CONTINUE"
        except Exception as e:
            print(f"Error draining node: {e}")
            result = "CONTINUE"  # don't block termination even if drain fails

        asg_client.complete_lifecycle_action(
            LifecycleHookName=lifecycle_hook_name,
            AutoScalingGroupName=asg_name,
            LifecycleActionResult=result,
            InstanceId=instance_id,
        )
