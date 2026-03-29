output "cluster_name"       { value = module.eks.cluster_name }
output "cluster_endpoint"   { value = module.eks.cluster_endpoint }
output "kubeconfig_command" { value = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}" }
output "vpc_id"             { value = module.vpc.vpc_id }
output "nlb_elastic_ips"    { value = aws_eip.nlb_eip[*].public_ip }
output "alb_controller_role_arn" { value = aws_iam_role.alb_controller.arn }
