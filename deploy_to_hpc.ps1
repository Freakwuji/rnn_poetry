# PowerShell 脚本：部署 RNN 项目到武大超算并提交训练
# 用法：右键 → 使用 PowerShell 运行

$hostname = "login.hpc.whu.edu.cn"
$username = $env:HPC_USER          # 从环境变量读，勿把账号写进仓库
$localPath = $PSScriptRoot        # 脚本所在目录

if (-not $username) {
    Write-Host "请先设置环境变量 HPC_USER（你的超算账号）" -ForegroundColor Red
    exit 1
}
$remotePath = "~/rnn_poetry"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "   部署 RNN 古诗词项目到武大超算" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 1. 打包项目
Write-Host "[1/4] 打包项目文件..." -ForegroundColor Yellow
$tarFile = "$env:TEMP\rnn_poetry.tar.gz"
if (Test-Path $tarFile) { Remove-Item $tarFile }

# 创建临时打包目录
$tmpDir = "$env:TEMP\rnn_poetry_pkg"
if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }
New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

# 复制需要的文件
Copy-Item "$localPath\*.py" $tmpDir
Copy-Item "$localPath\*.txt" $tmpDir
Copy-Item "$localPath\data\*" "$tmpDir\data\" -Recurse -Force
New-Item -ItemType Directory -Path "$tmpDir\checkpoints" -Force | Out-Null

# 使用 tar（Windows 10 1803+ 自带）
tar -czf $tarFile -C $tmpDir .
Write-Host "   打包完成: $(Get-Item $tarFile).Length / 1KB KB"

# 2. 测试 SSH 连接
Write-Host "[2/4] 测试 SSH 连接..." -ForegroundColor Yellow
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 $username@$hostname "hostname; whoami" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "   SSH 连接失败！请检查网络和账号" -ForegroundColor Red
    Write-Host "   手动连接: ssh $username@$hostname" -ForegroundColor Gray
    exit 1
}
Write-Host "   SSH 连接成功！" -ForegroundColor Green

# 3. 上传文件
Write-Host "[3/4] 上传项目文件..." -ForegroundColor Yellow
scp -q $tarFile $username@$hostname:~/rnn_poetry.tar.gz
if ($LASTEXITCODE -eq 0) {
    Write-Host "   上传成功！" -ForegroundColor Green
} else {
    Write-Host "   上传失败！" -ForegroundColor Red
    exit 1
}

# 4. 远程部署并提交作业
Write-Host "[4/4] 远程部署并提交 Slurm 作业..." -ForegroundColor Yellow

$remoteCommands = @"
set -e
echo "  连接成功，开始部署..."
cd ~
rm -rf rnn_poetry
mkdir -p rnn_poetry
cd rnn_poetry
tar xzf ../rnn_poetry.tar.gz
rm ../rnn_poetry.tar.gz
echo "  文件解压完成"

# 检查数据
echo "  数据: `$(wc -l < data/poetry.txt) 首古诗"
echo "  词表: `$(python3 -c "import json; v=json.load(open('data/vocab.json')); print(v['size'], 'chars')" 2>/dev/null || echo '需重建')"

# 提交 Slurm 作业
echo "  提交 Slurm 作业..."
JOB_ID=`$(sbatch --parsable hpc_job.sh 2>/dev/null || echo "none")`
if [ "$JOB_ID" != "none" ]; then
    echo "  作业已提交！JOB ID: `$JOB_ID"
    echo "  查看状态: squeue -j `$JOB_ID"
else
    echo "  尝试直接运行（slurm不可用）..."
    nohup python main.py --train > direct_run.log 2>&1 &
    echo "  后台运行中，PID: `$!"
fi
"@

ssh -o StrictHostKeyChecking=accept-new $username@$hostname $remoteCommands

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "部署完成！" -ForegroundColor Green
Write-Host "查看作业状态:" -ForegroundColor Gray
Write-Host "  ssh $username@$hostname" -ForegroundColor Gray
Write-Host "  squeue -u $username" -ForegroundColor Gray
Write-Host "  cat ~/rnn_poetry/job_*.log" -ForegroundColor Gray
Write-Host "======================================" -ForegroundColor Cyan

# 清理
Remove-Item $tarFile -Force
Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
