文件说明：
/src/poems_preprocess.py    用来进行语料的预处理，将语料转化为需要的形式
/src/models.py    定义poemRNN类，用来训练和预测诗句
/src/poems_generate.py    定义poemGenerator类，处理特定的诗句，调用poemRNN类来训练，最后根据首字生成唐诗
/corpus/    存放语料和一些中间文件
/tmp/    存放模型的训练结果

使用说明：
切换到/src/目录下，执行
1.python poems_preprocess.py（必须先进行的，只需要运行一次即可）
2.python poems_generator.py -m '../tmp/tmp17123101/' -e 60 -l 32
-m后面跟着的是训练模型的存放路径（checkpoint），默认是'../tmp/'，使用时需保证该目录存在
-e后接RNN模型迭代的次数，默认是50
-l后接制定的唐诗的长度，后面的唐诗的训练还有生成都由它来确定长度，默认是24（即五绝）
