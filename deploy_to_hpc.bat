@echo off
chcp 65001 >nul
echo ======================================
echo   部署 RNN 项目到武大超算
echo ======================================
echo.

set HOST=login.hpc.whu.edu.cn
set USER=clli_04

:: 第1步：打包项目
echo [1/4] 打包项目文件...
cd /d "D:\My World\Study\Program Practise\Python\rnn_poetry"
tar -czf %TEMP%\rnn_poetry.tar.gz *.py *.txt data\ checkpoints\ 2>nul
echo    完成
echo.

:: 第2步：上传文件
echo [2/4] 上传到超算（请输入密码: clli@123$）
echo.
scp %TEMP%\rnn_poetry.tar.gz %USER%@%HOST%:~/
echo.
if %ERRORLEVEL% neq 0 (
    echo    上传失败！请检查网络和密码
    pause
    exit /b 1
)
echo    上传成功！
echo.

:: 第3步：远程部署
echo [3/4] 部署到远程服务器...
ssh %USER%@%HOST% "cd ~ && rm -rf rnn_poetry && mkdir rnn_poetry && cd rnn_poetry && tar xzf ../rnn_poetry.tar.gz && rm ../rnn_poetry.tar.gz && echo 部署完成"
echo.

:: 第4步：提交训练作业
echo [4/4] 提交 Slurm 训练作业...
echo.
echo 正在连接超算提交作业...
echo 请输入密码（同上）
echo.
ssh %USER%@%HOST% "cd ~/rnn_poetry && sbatch hpc_job.sh"
echo.
echo ======================================
echo 部署流程完成！
echo.
echo 查看作业状态:
echo   ssh %USER%@%HOST% "squeue -u %USER%"
echo.
echo 查看训练日志:
echo   ssh %USER%@%HOST% "cat ~/rnn_poetry/job_*.log"
echo.
echo 下载训练好的模型:
echo   scp %USER%@%HOST%:~/rnn_poetry/checkpoints/lstm_poetry_best.pt D:\
echo.======================================
pause
