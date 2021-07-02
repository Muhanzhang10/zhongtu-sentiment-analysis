# zhongtu-sentiment-analysis

运行“sent.py”，测试程序的可运行性（第一步）。所有程序运行时需要打开VPN

运行“main.py” ，程序会从data文件夹中的文件读取，将结果写入output文件夹中（读取文件可以在程序标注处更改）

运行“main_oversea.py”，程序会从data文件夹中的文件读取，将部分结果写入output文件夹中（需要分析的图书可以在程序标注处通过行数进行更改。由于海外数据量庞大，通过此可以分批次进行计算）

运行"category.py"，程序会从data文件夹中读取，截取部分行数（可以自行调整）再存入data文件夹中

由于excel文件过大，github中不能存储，因此“海外”的数据需要从本地拷贝

output文件夹中的内容为运行后例子，可忽视
