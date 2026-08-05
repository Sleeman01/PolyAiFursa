output "control_plane_public_ip" {
  value = aws_instance.control_plane.public_ip
}

output "control_plane_private_ip" {
  value = aws_instance.control_plane.private_ip
}

output "control_plane_instance_id" {
  value = aws_instance.control_plane.id
}

output "asg_name" {
  value = aws_autoscaling_group.workers.name
}

output "cluster_sg_id" {
  value = aws_security_group.cluster_sg.id
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts_topic.arn
}
