# deploy/vastai.tf
terraform {
  required_providers {
    vastai = {
      source  = "vast-ai/vastai"
      version = "~> 1.0"
    }
  }
}

provider "vastai" {
  api_key = var.vastai_api_key
}

# GPU实例配置 - 量子奇点狙击系统优化
resource "vastai_instance" "quantum_sniper_gpu" {
  # GPU配置
  gpu_name     = "RTX 4090"
  gpu_count    = 1
  gpu_ram      = 24576  # 24GB
  
  # 计算资源配置
  cpu_cores    = 8
  ram          = 32768  # 32GB
  disk_size    = 100    # 100GB SSD
  
  # 镜像和启动配置
  image        = "nvidia/cuda:11.8-runtime-ubuntu20.04"
  onstart      = file("${path.module}/start_gpu_instance.sh")
  
  # 网络配置
  ssh_public_key = file("~/.ssh/id_rsa.pub")
  
  # 环境变量
  env = {
    ENVIRONMENT          = "gpu-production"
    QUANTUM_SYSTEM       = "v4.2"
    GPU_ENABLED          = "true"
    CUDA_VISIBLE_DEVICES = "0"
  }
  
  # 标签
  labels = {
    project     = "quantum-sniper"
    version     = "4.2"
    environment = "production"
    managed_by  = "terraform"
  }
}

# 输出实例信息
output "instance_info" {
  value = {
    id         = vastai_instance.quantum_sniper_gpu.id
    ip_address = vastai_instance.quantum_sniper_gpu.public_ip
    status     = vastai_instance.quantum_sniper_gpu.status
    ssh_command = "ssh root@${vastai_instance.quantum_sniper_gpu.public_ip}"
  }
}

# 变量定义
variable "vastai_api_key" {
  description = "VAST.ai API密钥"
  type        = string
  sensitive   = true
}

variable "quantum_version" {
  description = "量子系统版本"
  type        = string
  default     = "4.2"
}