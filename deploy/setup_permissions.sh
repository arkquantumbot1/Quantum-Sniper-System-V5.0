#!/bin/bash
# deploy/setup_permissions.sh
echo "🔒 设置部署脚本权限..."
chmod +x deploy/start_gpu_instance.sh
chmod +x deploy/stop_gpu_instance.sh  
chmod +x setup_production_env.sh
chmod +x deploy/setup_permissions.sh
echo "✅ 权限设置完成"