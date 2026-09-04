# 测试说明

运行仓库中的确定性测试：

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

这些测试不需要API密钥，也不会向外部服务发送案件数据。依赖私有生产数据的测试在相应文件不存在时自动跳过。

