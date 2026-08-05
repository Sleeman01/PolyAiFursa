data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node_role" {
  name               = "sleeman-${var.cluster_name}-node-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_iam_role_policy_attachment" "ebs_csi_policy" {
  role       = aws_iam_role.node_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

resource "aws_iam_role_policy_attachment" "ecr_readonly_policy" {
  role       = aws_iam_role.node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "node_profile" {
  name = "sleeman-${var.cluster_name}-node-profile"
  role = aws_iam_role.node_role.name
}

resource "aws_security_group" "cluster_sg" {
  name        = "sleeman-${var.cluster_name}-sg"
  description = "Allow SSH and all intra-VPC traffic"
  vpc_id      = var.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Intra-VPC traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    description = "Kubernetes API server"
    from_port   = 6443
    to_port     = 6443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "sleeman-${var.cluster_name}-sg"
  }

  lifecycle {
    ignore_changes = [ingress]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

resource "aws_iam_role_policy" "ssm_join_command" {
  name = "sleeman-${var.cluster_name}-ssm-join"
  role = aws_iam_role.node_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ssm:PutParameter", "ssm:GetParameter"]
        Resource = "arn:aws:ssm:*:*:parameter/sleeman/${var.cluster_name}/*"
      }
    ]
  })
}

resource "aws_instance" "control_plane" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = "t3.medium"
  key_name               = var.instance_key_name
  subnet_id              = var.public_subnet_ids[0]
  vpc_security_group_ids = [aws_security_group.cluster_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.node_profile.name

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/scripts/control-plane-init.sh.tftpl", {
    region              = var.region
    cluster_name        = var.cluster_name
    cert_refresh_script = file("${path.module}/scripts/cert-refresh-per-boot.sh.tftpl")
  })

  tags = {
    Name = "sleeman-${var.cluster_name}-control-plane"
    Role = "control-plane"
  }
}

resource "aws_launch_template" "worker" {
  name_prefix   = "sleeman-${var.cluster_name}-worker-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = "t3.medium"
  key_name      = var.instance_key_name

  iam_instance_profile {
    name = aws_iam_instance_profile.node_profile.name
  }

  vpc_security_group_ids = [aws_security_group.cluster_sg.id]

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 20
      volume_type = "gp3"
    }
  }

  user_data = base64encode(templatefile("${path.module}/scripts/worker-init.sh.tftpl", {
    region       = var.region
    cluster_name = var.cluster_name
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "sleeman-${var.cluster_name}-worker"
      Role = "worker"
    }
  }
}

resource "aws_autoscaling_group" "workers" {
  name                = "sleeman-${var.cluster_name}-workers-asg"
  vpc_zone_identifier = var.public_subnet_ids
  min_size            = 1
  max_size            = 3
  desired_capacity    = 2

  launch_template {
    id      = aws_launch_template.worker.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "sleeman-${var.cluster_name}-worker"
    propagate_at_launch = true
  }
}

# --- SNS topic for ASG lifecycle events ---
resource "aws_sns_topic" "lifecycle_topic" {
  name = "sleeman-${var.cluster_name}-lifecycle-topic"
}

# --- SNS topic for Alertmanager notifications ---
resource "aws_sns_topic" "alerts_topic" {
  name = "sleeman-${var.cluster_name}-alerts-topic"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts_topic.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# --- Allow cluster nodes to publish to the alerts topic. No IRSA on this
# self-managed cluster; Alertmanager's sigv4 auth falls back to the
# instance's IAM role via node_role/node_profile. ---
resource "aws_iam_role_policy" "node_sns_publish" {
  name = "sleeman-${var.cluster_name}-node-sns-publish"
  role = aws_iam_role.node_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.alerts_topic.arn
      }
    ]
  })
}

# --- IAM role for Lambda ---
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_role" {
  name               = "sleeman-${var.cluster_name}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "sleeman-${var.cluster_name}-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["autoscaling:CompleteLifecycleAction"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = "arn:aws:secretsmanager:*:*:secret:sleeman-t006-k8s-cp-ssh-key*"
      },
      {
        Effect   = "Allow"
        Action   = ["ec2:DescribeInstances"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# --- Lambda function ---
resource "aws_lambda_function" "node_drain" {
  function_name = "sleeman-${var.cluster_name}-node-drain"
  role          = aws_iam_role.lambda_role.arn
  handler       = "node_drain.handler"
  runtime       = "python3.12"
  timeout       = 30
  filename      = "${path.module}/lambda/node_drain.zip"
  source_code_hash = filebase64sha256("${path.module}/lambda/node_drain.zip")

  environment {
    variables = {
      SSH_SECRET_NAME        = "sleeman-t006-k8s-cp-ssh-key"
      CONTROL_PLANE_PUBLIC_IP = aws_instance.control_plane.public_ip
    }
  }
}

resource "aws_lambda_permission" "allow_sns" {
  statement_id  = "AllowSNSInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.node_drain.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.lifecycle_topic.arn
}

resource "aws_sns_topic_subscription" "lambda_sub" {
  topic_arn = aws_sns_topic.lifecycle_topic.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.node_drain.arn
}

# --- IAM role allowing ASG to publish to SNS ---
data "aws_iam_policy_document" "asg_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["autoscaling.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "asg_lifecycle_role" {
  name               = "sleeman-${var.cluster_name}-asg-lifecycle-role"
  assume_role_policy = data.aws_iam_policy_document.asg_assume_role.json
}

resource "aws_iam_role_policy" "asg_sns_publish" {
  name = "sleeman-${var.cluster_name}-asg-sns-publish"
  role = aws_iam_role.asg_lifecycle_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.lifecycle_topic.arn
      }
    ]
  })
}

# --- ASG lifecycle hook on termination ---
resource "aws_autoscaling_lifecycle_hook" "drain_on_terminate" {
  name                   = "sleeman-${var.cluster_name}-drain-hook"
  autoscaling_group_name = aws_autoscaling_group.workers.name
  lifecycle_transition   = "autoscaling:EC2_INSTANCE_TERMINATING"
  default_result         = "CONTINUE"
  heartbeat_timeout      = 120
  notification_target_arn = aws_sns_topic.lifecycle_topic.arn
  role_arn               = aws_iam_role.asg_lifecycle_role.arn
}

