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
