output "control_plane_public_ip" {
  value = module.k8s_cluster.control_plane_public_ip
}

output "control_plane_private_ip" {
  value = module.k8s_cluster.control_plane_private_ip
}

output "asg_name" {
  value = module.k8s_cluster.asg_name
}
