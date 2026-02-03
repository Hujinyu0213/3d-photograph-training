pointnet\_regression\_model\_full    error is really big(xunlian jieguo fengxi.py)



1/8

k fold consequence:



✅ 成功加载 100 个样本

   点云数量统计:

     最小: 11564 个点

     最大: 29182 个点

     平均: 19345 个点



统一采样到 8192 个点...

   最终数据形状: X=(100, 3, 8192), Y=(100, 27)



============================================================

🔄 K折交叉验证训练 (K=5) + 独立测试集

============================================================

总样本数: 100



📊 数据划分:

   测试集: 10 个样本 (10%)

   训练+验证集: 90 个样本 (90%)

   每折验证集大小: 约 18 个样本

   每折训练集大小: 约 72 个样本



============================================================

📊 折 1/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 5059616.083333 | Val Loss: 5025350.000000 | Best Val: 5025350.000000

  Epoch   50/500 | Train Loss: 4305187.305556 | Val Loss: 4459212.583333 | Best Val: 4459212.583333

  Epoch  100/500 | Train Loss: 2780444.500000 | Val Loss: 3314658.000000 | Best Val: 3247990.666667

  Epoch  150/500 | Train Loss: 1303197.652778 | Val Loss: 1834445.708333 | Best Val: 1755061.833333

  Epoch  200/500 | Train Loss: 780992.631944 | Val Loss: 1365052.416667 | Best Val: 1272612.250000

  Epoch  250/500 | Train Loss: 392176.338542 | Val Loss: 852062.625000 | Best Val: 852062.625000

  Epoch  300/500 | Train Loss: 246570.492622 | Val Loss: 560720.104167 | Best Val: 560720.104167

  Epoch  350/500 | Train Loss: 132765.501845 | Val Loss: 613898.708333 | Best Val: 494825.604167

  Epoch  400/500 | Train Loss: 82749.769206 | Val Loss: 466871.843750 | Best Val: 349235.333333

  Epoch  450/500 | Train Loss: 40758.305990 | Val Loss: 372923.291667 | Best Val: 311101.770833

  Epoch  500/500 | Train Loss: 54166.776150 | Val Loss: 330667.958333 | Best Val: 309707.864583



✅ 折 1 训练完成

   最佳验证损失: 309707.864583

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold1\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold1\_final.pth



============================================================

📊 折 2/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 5155512.861111 | Val Loss: 4613123.000000 | Best Val: 4613123.000000

  Epoch   50/500 | Train Loss: 4371430.777778 | Val Loss: 4075964.583333 | Best Val: 4065407.166667

  Epoch  100/500 | Train Loss: 2806836.805556 | Val Loss: 2992764.500000 | Best Val: 2784592.333333

  Epoch  150/500 | Train Loss: 1351549.833333 | Val Loss: 1687476.291667 | Best Val: 1637971.333333

  Epoch  200/500 | Train Loss: 779599.781250 | Val Loss: 1275611.791667 | Best Val: 1200375.333333

  Epoch  250/500 | Train Loss: 436540.477431 | Val Loss: 920607.937500 | Best Val: 491636.812500

  Epoch  300/500 | Train Loss: 201388.919271 | Val Loss: 356928.994792 | Best Val: 302833.908854

  Epoch  350/500 | Train Loss: 117594.917969 | Val Loss: 243613.236979 | Best Val: 243613.236979

  Epoch  400/500 | Train Loss: 94877.417969 | Val Loss: 216098.390625 | Best Val: 208034.269531

  Epoch  450/500 | Train Loss: 59606.159559 | Val Loss: 161674.391276 | Best Val: 151047.169922

  Epoch  500/500 | Train Loss: 46397.759115 | Val Loss: 139158.537109 | Best Val: 138477.269857



✅ 折 2 训练完成

   最佳验证损失: 138477.269857

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold2\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold2\_final.pth



============================================================

📊 折 3/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 5108278.833333 | Val Loss: 4925377.500000 | Best Val: 4925377.500000

  Epoch   50/500 | Train Loss: 4351166.361111 | Val Loss: 4364839.666667 | Best Val: 4364839.666667

  Epoch  100/500 | Train Loss: 2835874.527778 | Val Loss: 3061105.166667 | Best Val: 2901718.833333

  Epoch  150/500 | Train Loss: 1355655.881944 | Val Loss: 1691649.125000 | Best Val: 1625323.333333

  Epoch  200/500 | Train Loss: 828603.197917 | Val Loss: 1074168.270833 | Best Val: 1053303.062500

  Epoch  250/500 | Train Loss: 414022.076389 | Val Loss: 691834.104167 | Best Val: 671073.520833

  Epoch  300/500 | Train Loss: 224496.109375 | Val Loss: 492103.718750 | Best Val: 315487.875000

  Epoch  350/500 | Train Loss: 134233.950738 | Val Loss: 313109.218750 | Best Val: 302867.895833

  Epoch  400/500 | Train Loss: 150092.672309 | Val Loss: 177192.296875 | Best Val: 74615.360677

  Epoch  450/500 | Train Loss: 101446.566623 | Val Loss: 116334.136963 | Best Val: 17056.982096

  Epoch  500/500 | Train Loss: 40622.528212 | Val Loss: 80002.113281 | Best Val: 17056.982096



✅ 折 3 训练完成

   最佳验证损失: 17056.982096

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold3\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold3\_final.pth



============================================================

📊 折 4/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 5138425.805556 | Val Loss: 4526014.416667 | Best Val: 4526014.416667

  Epoch   50/500 | Train Loss: 4368666.305556 | Val Loss: 4058652.916667 | Best Val: 3952837.416667

  Epoch  100/500 | Train Loss: 2857263.388889 | Val Loss: 2882956.666667 | Best Val: 2617649.166667

  Epoch  150/500 | Train Loss: 1384509.069444 | Val Loss: 1818760.250000 | Best Val: 1719974.875000

  Epoch  200/500 | Train Loss: 818532.680556 | Val Loss: 971740.041667 | Best Val: 971740.041667

  Epoch  250/500 | Train Loss: 428907.477431 | Val Loss: 645578.208333 | Best Val: 557442.208333

  Epoch  300/500 | Train Loss: 216831.197917 | Val Loss: 425140.187500 | Best Val: 417623.916667

  Epoch  350/500 | Train Loss: 162430.115126 | Val Loss: 379044.916667 | Best Val: 215739.028646

  Epoch  400/500 | Train Loss: 95760.414062 | Val Loss: 343214.135417 | Best Val: 215739.028646

  Epoch  450/500 | Train Loss: 53529.644097 | Val Loss: 273058.114583 | Best Val: 131976.213542

  Epoch  500/500 | Train Loss: 80632.330838 | Val Loss: 205882.062500 | Best Val: 121692.210938



✅ 折 4 训练完成

   最佳验证损失: 121692.210938

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold4\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold4\_final.pth



============================================================

📊 折 5/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 4888534.222222 | Val Loss: 5721875.666667 | Best Val: 5721875.666667

  Epoch   50/500 | Train Loss: 4121292.333333 | Val Loss: 4836009.666667 | Best Val: 4772871.666667

  Epoch  100/500 | Train Loss: 2622034.583333 | Val Loss: 3088022.916667 | Best Val: 2878634.916667

  Epoch  150/500 | Train Loss: 1201203.729167 | Val Loss: 1555742.000000 | Best Val: 1555742.000000

  Epoch  200/500 | Train Loss: 754562.909722 | Val Loss: 1085259.604167 | Best Val: 1018834.666667

  Epoch  250/500 | Train Loss: 376076.430556 | Val Loss: 698952.552083 | Best Val: 654465.541667

  Epoch  300/500 | Train Loss: 213664.234375 | Val Loss: 418632.906250 | Best Val: 418632.906250

  Epoch  350/500 | Train Loss: 109294.053494 | Val Loss: 299852.557292 | Best Val: 299852.557292

  Epoch  400/500 | Train Loss: 79243.987413 | Val Loss: 257643.544271 | Best Val: 238065.945312

  Epoch  450/500 | Train Loss: 59030.371745 | Val Loss: 233254.420573 | Best Val: 192655.412760

  Epoch  500/500 | Train Loss: 41055.821181 | Val Loss: 197685.467448 | Best Val: 155257.381510



✅ 折 5 训练完成

   最佳验证损失: 155257.381510

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold5\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold5\_final.pth



============================================================

📊 K折交叉验证结果总结

============================================================

折 1: 最佳验证损失 = 309707.864583

折 2: 最佳验证损失 = 138477.269857

折 3: 最佳验证损失 = 17056.982096

折 4: 最佳验证损失 = 121692.210938

折 5: 最佳验证损失 = 155257.381510



统计信息:

  平均验证损失: 148438.341797 ± 93946.791781

  最小验证损失: 17056.982096 (折 3)

  标准差: 93946.791781



✅ 最佳模型已复制:

   来源: 折 3 的最佳模型

   目标: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_best.pth

   训练历史: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\training\_history\_kfold.json



============================================================

🔄 用所有训练+验证数据重新训练最终模型

============================================================

训练数据: 90 个样本（所有90%的数据）

目的: 充分利用所有数据，获得更好的最终模型



开始训练最终模型（500 轮）...

  Epoch    1/500 | Train Loss: 5060128.272727

  Epoch   50/500 | Train Loss: 4008477.522727

  Epoch  100/500 | Train Loss: 2060281.761364

  Epoch  150/500 | Train Loss: 701972.313920

  Epoch  200/500 | Train Loss: 298984.352983

  Epoch  250/500 | Train Loss: 131214.746715

  Epoch  300/500 | Train Loss: 33063.595614

  Epoch  350/500 | Train Loss: 24442.493519

  Epoch  400/500 | Train Loss: 70880.123890

  Epoch  450/500 | Train Loss: 41276.681996

  Epoch  500/500 | Train Loss: 38481.908825



✅ 最终模型训练完成！

   模型已保存: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_best.pth

   使用数据: 90 个样本（所有训练+验证数据）



============================================================

🧪 在测试集上评估最终模型

============================================================

测试集大小: 10 个样本

测试集损失: 244681.281250



🎉 K折交叉验证 + 最终模型训练完成！

   K折交叉验证: 训练了 5 个模型用于选择最佳配置

   平均验证损失: 148438.341797 ± 93946.791781

   最佳折: 折 3，验证损失: 17056.982096

   最终模型: 用所有 90 个样本重新训练

   ⭐ 测试集损失: 244681.281250 (独立评估，无偏)



📁 最终模型文件: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_best.pth











80/20 split   update normalization





✅ 成功加载 100 个样本

   点云数量统计:

     最小: 11564 个点

     最大: 29182 个点

     平均: 19345 个点



统一采样到 8192 个点...

   最终数据形状: X=(100, 3, 8192), Y=(100, 27)



📊 数据集划分: 训练集 80 个样本, 验证集 20 个样本



--- 开始训练 500 轮 ---

📋 训练配置:

   批次大小: 8

   初始学习率: 0.001

   学习率衰减: 每 150 轮 × 0.5

   Dropout: 0.3

   特征变换正则化权重: 0.001

Epoch    1/500 | Train Loss: 0.235988 | Val Loss: 0.016592 | LR: 0.001000 | Best Val: 0.016592

Epoch   25/500 | Train Loss: 0.012400 | Val Loss: 0.024734 | LR: 0.001000 | Best Val: 0.011475

Epoch   50/500 | Train Loss: 0.004538 | Val Loss: 0.006068 | LR: 0.001000 | Best Val: 0.002903

Epoch   75/500 | Train Loss: 0.001806 | Val Loss: 0.006067 | LR: 0.001000 | Best Val: 0.001367

Epoch  100/500 | Train Loss: 0.000655 | Val Loss: 0.020283 | LR: 0.001000 | Best Val: 0.001367

Epoch  125/500 | Train Loss: 0.000607 | Val Loss: 0.008858 | LR: 0.001000 | Best Val: 0.001367

Epoch  150/500 | Train Loss: 0.000303 | Val Loss: 0.002547 | LR: 0.000500 | Best Val: 0.000419

Epoch  175/500 | Train Loss: 0.000178 | Val Loss: 0.002607 | LR: 0.000500 | Best Val: 0.000399

Epoch  200/500 | Train Loss: 0.000148 | Val Loss: 0.008583 | LR: 0.000500 | Best Val: 0.000399

Epoch  225/500 | Train Loss: 0.000304 | Val Loss: 0.004597 | LR: 0.000500 | Best Val: 0.000399

Epoch  250/500 | Train Loss: 0.000199 | Val Loss: 0.019944 | LR: 0.000500 | Best Val: 0.000399

Epoch  275/500 | Train Loss: 0.000121 | Val Loss: 0.000294 | LR: 0.000500 | Best Val: 0.000294

Epoch  300/500 | Train Loss: 0.000226 | Val Loss: 0.000269 | LR: 0.000250 | Best Val: 0.000140

Epoch  325/500 | Train Loss: 0.000176 | Val Loss: 0.000177 | LR: 0.000250 | Best Val: 0.000140

Epoch  350/500 | Train Loss: 0.000184 | Val Loss: 0.000167 | LR: 0.000250 | Best Val: 0.000111

Epoch  375/500 | Train Loss: 0.000154 | Val Loss: 0.000238 | LR: 0.000250 | Best Val: 0.000106

Epoch  400/500 | Train Loss: 0.000121 | Val Loss: 0.000088 | LR: 0.000250 | Best Val: 0.000082

Epoch  425/500 | Train Loss: 0.000139 | Val Loss: 0.000178 | LR: 0.000250 | Best Val: 0.000082

Epoch  450/500 | Train Loss: 0.000161 | Val Loss: 0.000142 | LR: 0.000125 | Best Val: 0.000082

Epoch  475/500 | Train Loss: 0.000197 | Val Loss: 0.000135 | LR: 0.000125 | Best Val: 0.000068

Epoch  500/500 | Train Loss: 0.000090 | Val Loss: 0.000067 | LR: 0.000125 | Best Val: 0.000066



🎉 训练完成！

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_full.pth

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_full\_best.pth (验证损失: 0.000066)

   训练历史: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\training\_history\_full.json











k fold:9th Jan







✅ 成功加载 100 个样本

   点云数量统计:

     最小: 11564 个点

     最大: 29182 个点

     平均: 19345 个点



统一采样到 8192 个点...

   最终数据形状: X=(100, 3, 8192), Y=(100, 27)



============================================================

🔄 K折交叉验证训练 (K=5) + 独立测试集

============================================================

总样本数: 100



📊 数据划分:

   测试集: 10 个样本 (10%)

   训练+验证集: 90 个样本 (90%)

   每折验证集大小: 约 18 个样本

   每折训练集大小: 约 72 个样本



============================================================

📊 折 1/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 0.240848 | Val Loss: 0.059467 | Best Val: 0.059467

  Epoch   50/500 | Train Loss: 0.005909 | Val Loss: 0.002824 | Best Val: 0.002824

  Epoch  100/500 | Train Loss: 0.001602 | Val Loss: 0.001024 | Best Val: 0.001018

  Epoch  150/500 | Train Loss: 0.000649 | Val Loss: 0.000598 | Best Val: 0.000538

  Epoch  200/500 | Train Loss: 0.000448 | Val Loss: 0.000374 | Best Val: 0.000330

  Epoch  250/500 | Train Loss: 0.000272 | Val Loss: 0.000289 | Best Val: 0.000268

  Epoch  300/500 | Train Loss: 0.000305 | Val Loss: 0.000414 | Best Val: 0.000259

  Epoch  350/500 | Train Loss: 0.000223 | Val Loss: 0.000225 | Best Val: 0.000172

  Epoch  400/500 | Train Loss: 0.000229 | Val Loss: 0.000229 | Best Val: 0.000172

  Epoch  450/500 | Train Loss: 0.000173 | Val Loss: 0.000251 | Best Val: 0.000168

  Epoch  500/500 | Train Loss: 0.000140 | Val Loss: 0.000191 | Best Val: 0.000136



✅ 折 1 训练完成

   最佳验证损失: 0.000136

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold1\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold1\_final.pth



============================================================

📊 折 2/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 0.233470 | Val Loss: 0.112948 | Best Val: 0.112948

  Epoch   50/500 | Train Loss: 0.004619 | Val Loss: 0.002343 | Best Val: 0.002330

  Epoch  100/500 | Train Loss: 0.001173 | Val Loss: 0.000825 | Best Val: 0.000825

  Epoch  150/500 | Train Loss: 0.000515 | Val Loss: 0.000949 | Best Val: 0.000475

  Epoch  200/500 | Train Loss: 0.000263 | Val Loss: 0.000653 | Best Val: 0.000329

  Epoch  250/500 | Train Loss: 0.000283 | Val Loss: 0.000596 | Best Val: 0.000329

  Epoch  300/500 | Train Loss: 0.000199 | Val Loss: 0.000712 | Best Val: 0.000329

  Epoch  350/500 | Train Loss: 0.000161 | Val Loss: 0.000546 | Best Val: 0.000288

  Epoch  400/500 | Train Loss: 0.000162 | Val Loss: 0.000671 | Best Val: 0.000288

  Epoch  450/500 | Train Loss: 0.000142 | Val Loss: 0.000605 | Best Val: 0.000288

  Epoch  500/500 | Train Loss: 0.000079 | Val Loss: 0.000805 | Best Val: 0.000288



✅ 折 2 训练完成

   最佳验证损失: 0.000288

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold2\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold2\_final.pth



============================================================

📊 折 3/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 0.259036 | Val Loss: 0.060156 | Best Val: 0.060156

  Epoch   50/500 | Train Loss: 0.006531 | Val Loss: 0.003156 | Best Val: 0.003156

  Epoch  100/500 | Train Loss: 0.001765 | Val Loss: 0.001530 | Best Val: 0.001308

  Epoch  150/500 | Train Loss: 0.000901 | Val Loss: 0.001149 | Best Val: 0.000869

  Epoch  200/500 | Train Loss: 0.000510 | Val Loss: 0.001021 | Best Val: 0.000756

  Epoch  250/500 | Train Loss: 0.000513 | Val Loss: 0.001887 | Best Val: 0.000688

  Epoch  300/500 | Train Loss: 0.000444 | Val Loss: 0.000788 | Best Val: 0.000687

  Epoch  350/500 | Train Loss: 0.000400 | Val Loss: 0.001369 | Best Val: 0.000674

  Epoch  400/500 | Train Loss: 0.000373 | Val Loss: 0.001123 | Best Val: 0.000615

  Epoch  450/500 | Train Loss: 0.000330 | Val Loss: 0.001148 | Best Val: 0.000436

  Epoch  500/500 | Train Loss: 0.000314 | Val Loss: 0.000921 | Best Val: 0.000436



✅ 折 3 训练完成

   最佳验证损失: 0.000436

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold3\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold3\_final.pth



============================================================

📊 折 4/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 0.236360 | Val Loss: 0.020317 | Best Val: 0.020317

  Epoch   50/500 | Train Loss: 0.006212 | Val Loss: 0.046898 | Best Val: 0.012983

  Epoch  100/500 | Train Loss: 0.001266 | Val Loss: 0.004980 | Best Val: 0.003041

  Epoch  150/500 | Train Loss: 0.000820 | Val Loss: 0.006302 | Best Val: 0.003041

  Epoch  200/500 | Train Loss: 0.000446 | Val Loss: 0.008386 | Best Val: 0.003041

  Epoch  250/500 | Train Loss: 0.000359 | Val Loss: 0.011522 | Best Val: 0.003041

  Epoch  300/500 | Train Loss: 0.000347 | Val Loss: 0.012666 | Best Val: 0.003041

  Epoch  350/500 | Train Loss: 0.000216 | Val Loss: 0.004725 | Best Val: 0.003041

  Epoch  400/500 | Train Loss: 0.000215 | Val Loss: 0.010047 | Best Val: 0.003041

  Epoch  450/500 | Train Loss: 0.000167 | Val Loss: 0.008227 | Best Val: 0.003041

  Epoch  500/500 | Train Loss: 0.000106 | Val Loss: 0.008779 | Best Val: 0.003041



✅ 折 4 训练完成

   最佳验证损失: 0.003041

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold4\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold4\_final.pth



============================================================

📊 折 5/5

============================================================

训练集: 72 个样本

验证集: 18 个样本

  Epoch    1/500 | Train Loss: 0.244226 | Val Loss: 0.048233 | Best Val: 0.048233

  Epoch   50/500 | Train Loss: 0.005745 | Val Loss: 0.006661 | Best Val: 0.005137

  Epoch  100/500 | Train Loss: 0.002461 | Val Loss: 0.006564 | Best Val: 0.004877

  Epoch  150/500 | Train Loss: 0.000618 | Val Loss: 0.001088 | Best Val: 0.000766

  Epoch  200/500 | Train Loss: 0.000323 | Val Loss: 0.001442 | Best Val: 0.000604

  Epoch  250/500 | Train Loss: 0.000370 | Val Loss: 0.001766 | Best Val: 0.000432

  Epoch  300/500 | Train Loss: 0.000255 | Val Loss: 0.000532 | Best Val: 0.000265

  Epoch  350/500 | Train Loss: 0.000217 | Val Loss: 0.000775 | Best Val: 0.000265

  Epoch  400/500 | Train Loss: 0.000146 | Val Loss: 0.002614 | Best Val: 0.000265

  Epoch  450/500 | Train Loss: 0.000138 | Val Loss: 0.003097 | Best Val: 0.000265

  Epoch  500/500 | Train Loss: 0.000113 | Val Loss: 0.001547 | Best Val: 0.000265



✅ 折 5 训练完成

   最佳验证损失: 0.000265

   最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold5\_best.pth

   最终模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_fold5\_final.pth



============================================================

📊 K折交叉验证结果总结

============================================================

折 1: 最佳验证损失 = 0.000136

折 2: 最佳验证损失 = 0.000288

折 3: 最佳验证损失 = 0.000436

折 4: 最佳验证损失 = 0.003041

折 5: 最佳验证损失 = 0.000265



统计信息:

  平均验证损失: 0.000833 ± 0.001108

  最小验证损失: 0.000136 (折 1)

  标准差: 0.001108



✅ 最佳模型已复制:

   来源: 折 1 的最佳模型

   目标: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_best.pth

   训练历史: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\training\_history\_kfold.json



============================================================

🔄 用所有训练+验证数据重新训练最终模型

============================================================

训练数据: 90 个样本（所有90%的数据）

目的: 充分利用所有数据，获得更好的最终模型



开始训练最终模型（500 轮）...

  Epoch    1/500 | Train Loss: 0.236645

  Epoch   50/500 | Train Loss: 0.004419

  Epoch  100/500 | Train Loss: 0.001090

  Epoch  150/500 | Train Loss: 0.000537

  Epoch  200/500 | Train Loss: 0.000357

  Epoch  250/500 | Train Loss: 0.000367

  Epoch  300/500 | Train Loss: 0.000278

  Epoch  350/500 | Train Loss: 0.000225

  Epoch  400/500 | Train Loss: 0.000239

  Epoch  450/500 | Train Loss: 0.000276

  Epoch  500/500 | Train Loss: 0.000182



✅ 最终模型训练完成！

   模型已保存: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_best.pth

   使用数据: 90 个样本（所有训练+验证数据）



============================================================

🧪 在测试集上评估最终模型

============================================================

测试集大小: 10 个样本

测试集损失: 0.002693



🎉 K折交叉验证 + 最终模型训练完成！

   K折交叉验证: 训练了 5 个模型用于选择最佳配置

   平均验证损失: 0.000833 ± 0.001108

   最佳折: 折 1，验证损失: 0.000136

   最终模型: 用所有 90 个样本重新训练

   ⭐ 测试集损失: 0.002693 (独立评估，无偏)



📁 最终模型文件: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\pointnet\_regression\_model\_kfold\_best.pth





pointnet\_regression\_model\_kfold\_best.pth



this is not the best one//// the best k fold should be 1th fold

i use 1th fold to test first, but i think maybe ze need to retrain on k fold









#### 2/2 note hu



**1. Tried to train with FPS sampling and add the validation set and data enhancing. The error is still huge compared to the ideal value 2mm, the full result compared to the kfold model are as below. The script is : scripts/training/main\_script\_full\_pointcloud\_aug\_fps.py**



============================================================

📊 测试集性能对比 (毫米)

============================================================



整体指标对比 (毫米):

指标                            增强+FPS            K折模型         改进

---

MSE(mm^2)                 161.196320      191.456848     15.81%

RMSE(mm)                   12.696311       13.836793      8.24%

MAE(mm)                    10.749276       10.996382      2.25%



每个地标点3D误差对比 (RMSE, mm):

地标点                   增强+FPS         K折模型         改进

---

Glabella           26.898201    17.035503    -57.89%

Nasion             19.451246    34.704235     43.95%

Rhinion            20.173586    26.513836     23.91%

Nasal Tip          14.824703    25.115013     40.97%

Subnasale          25.173956    12.684813    -98.46%

Alare (R)          18.334316    24.811174     26.10%

Alare (L)          23.452953    24.995451      6.17%

Zygion (R)         17.486124    18.085920      3.32%

Zygion (L)         28.249075    24.751223    -14.13%



📁 详细结果已保存: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\results\\test\_comparison\_aug\_fps\_vs\_old\_kfold.json



============================================================

💡 结论与建议

============================================================

✅ 增强+FPS模型略优于K折模型 (RMSE改进 8.24%)

   建议: 可以使用增强+FPS模型，或结合两者优点



下一步:

  1. 如果增强+FPS效果好，可以用它做K折交叉验证

  2. 尝试 PointNet++ 或其他架构

  3. 收集更多训练数据

  4. 调整增强参数以获得更好的泛化

============================================================



**2. Try to combine k-fold with all these changes above.**

**新 K折模型在部分点上有显著改进（Subnasale, Alare L, Zygion L），但在另一些点反而变差（Nasion, Zygion R）**

**整体 RMSE/MAE 略有改进，但不稳定，说明需要更多数据或进一步调优**

**距离 2mm 目标还很远（当前约 12-13mm），需要收集更多样本或尝试 PointNet++**



📦 K折训练完成

平均最佳验证损失: 0.0002540331498797362

最佳折: 1 loss: 0.00018740897454942265

最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\models\\pointnet\_regression\_model\_kfold\_aug\_fps\_best.pth

训练历史: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\results\\training\_history\_kfold\_aug\_fps.json







============================================================

📊 测试集性能对比 (毫米)

============================================================



整体指标对比 (毫米):

指标                            增强+FPS            K折模型         改进

---

MSE(mm^2)                 161.163361      158.556595     -1.64%

RMSE(mm)                   12.695013       12.591926     -0.82%

MAE(mm)                    10.748398        9.982676     -7.67%



每个地标点3D误差对比 (RMSE, mm):

地标点                   增强+FPS         K折模型         改进

---

Glabella           26.896006    25.893202     -3.87%

Nasion             19.449949    25.875690     24.83%

Rhinion            20.171515    22.846302     11.71%

Nasal Tip          14.824083    18.640568     20.47%

Subnasale          25.170589    17.676407    -42.40%

Alare (R)          18.332071    21.040945     12.87%

Alare (L)          23.449505    13.022996    -80.06%

Zygion (R)         17.482834    30.115946     41.95%

Zygion (L)         28.247475    15.487059    -82.39%



📁 详细结果已保存: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\results\\test\_comparison\_aug\_fps\_vs\_kfold\_aug\_fps.json



#### 2/3 note hu

1. **Ran hyperparameter tuning (random search, 20 trials) on K-fold + FPS + 增强.**
   Best params: LR 0.0015, BS 8, Dropout 0.35, Step 120, Gamma 0.7, FT 0.001.

* Validation mean loss improved 0.000254 → 0.00007065 (≈72%).
* Added scripts: main\_script\_tuned\_kfold.py (train with best params, 300 epochs, 5 folds), evaluate\_tuned\_model\_comprehensive.py (test compare vs all historical models), analyze\_hyperparameter\_tuning.py (summaries).
* Pending: run tuned training + comprehensive test evaluation to get final mm metrics.



**2. Train with tuned hyperparameter**



✨ 最佳折: Fold 4

   最佳验证损失: 0.00001178



📊 K 折统计:

   平均验证损失: 0.00001726 ± 0.00000279



💾 最佳模型已保存: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\models\\pointnet\_regression\_model\_tuned\_kfold\_best.pth

📁 训练历史已保存: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\results\\training\_histories\\training\_history\_tuned\_kfold.json

================================================================================



================================================================================

🧪 超参数调优模型 vs 所有历史模型 - 测试集对比

================================================================================



测试集样本数: 20



================================================================================

评估模型: 超参数调优K折模型

================================================================================

  RMSE: 13.3969 mm

  MAE:  11.2938 mm



================================================================================

评估模型: K折+增强+FPS

================================================================================

  RMSE: 12.5920 mm

  MAE:  9.9827 mm



================================================================================

评估模型: 单次训练+增强+FPS

================================================================================

  RMSE: 12.6953 mm

  MAE:  10.7486 mm



================================================================================

评估模型: 旧K折模型（无增强）

================================================================================

  RMSE: 13.8366 mm

  MAE:  10.9962 mm



================================================================================

📊 测试集性能对比 (毫米)

================================================================================



模型                                    RMSE(mm)         MAE(mm)          vs 旧K折

---

超参数调优K折模型                              13.3969         11.2938          +3.18%

K折+增强+FPS                              12.5920          9.9827          +8.99%

单次训练+增强+FPS                            12.6953         10.7486          +8.25%

旧K折模型（无增强）                             13.8366         10.9962          +0.00%



================================================================================

📍 每个地标点 RMSE 对比 (mm)

================================================================================



地标点                  超参数调优K折模型      K折+增强+FPS    单次训练+增强+FPS     旧K折模型（无增强）

---

Glabella               20.0195        25.8929        26.8965        17.0340

Nasion                 24.5034        25.8761        19.4501        34.7037

Rhinion                27.7863        22.8462        20.1718        26.5136

Nasal Tip              25.8702        18.6410        14.8243        25.1121

Subnasale              20.7761        17.6761        25.1705        12.6844

Alare (R)              17.4042        21.0409        18.3322        24.8110

Alare (L)              22.1051        13.0236        23.4497        24.9964

Zygion (R)             22.2331        30.1159        17.4840        18.0875

Zygion (L)             26.1885        15.4877        28.2487        24.7512



📁 详细结果已保存: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\results\\test\_evaluations\\test\_comparison\_all\_models\_with\_tuned.json



================================================================================

💡 结论

================================================================================



🎯 超参数调优模型测试集 RMSE: 13.3969 mm

   相比旧K折模型改进: 3.18%



⚠️  距离 2mm 目标仍有差距，建议:

   1. 收集更多训练样本

   2. 尝试 PointNet++ 架构

   3. 分析高误差地标点特征

================================================================================



**3. Train with pointnet++ : using 5-fold cross-validation with data augmentation and farthest-point sampling (8192 points), 90/10 of train/test split**



===== 在测试集上评估 =====

C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\scripts\\training\\main\_script\_pointnet2\_kfold.py:239: FutureWarning: You are using `torch.load` with `weights\_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights\_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add\_safe\_globals`. We recommend you start setting `weights\_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.

&nbsp; best\_model.load\_state\_dict(torch.load(best\_overall\_model))

测试集损失: 0.000025

L2距离 (平均): 0.008098 ± 0.003017

各地标L2距离: \[0.00828781 0.00799164 0.0060118  0.00537658 0.00875341 0.00886916

&nbsp;0.00870771 0.00947082 0.00941711]



==== 训练完成 ====

最佳折: Fold 5, 最佳验证损失: 0.000009

平均验证损失: 0.000012 ± 0.000002

测试集损失: 0.000025

最佳模型: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\models\\pointnet2\_regression\_kfold\_best.pth

训练历史: C:\\Users\\mkale\\Desktop\\Pointnet\_Pointnet2\_pytorch-master\\PointFeatureProject\\results\\training\_histories\\training\_history\_pointnet2\_kfold.json

================================================================================
EVALUATION RESULTS
================================================================================

Number of samples: 10
Number of landmarks: 9

--------------------------------------------------------------------------------
L2 DISTANCE - NORMALIZED SPACE
--------------------------------------------------------------------------------
  Mean: 0.008148
  Std:  0.002876
  Per-landmark:
    Landmark 1     : 0.008222
    Landmark 2     : 0.008244
    Landmark 3     : 0.006237
    Landmark 4     : 0.005251
    Landmark 5     : 0.008664
    Landmark 6     : 0.009041
    Landmark 7     : 0.008762
    Landmark 8     : 0.009527
    Landmark 9     : 0.009382



**4. PointNet++ Hyperparameter Search with Grid Search**

Created `scripts/training/hparam_search_pointnet2.py` to systematically search best hyperparameters for PointNet++ regression model.

📋 **Search Space:**
   - Learning Rate: [1e-3, 5e-4]
   - Dropout: [0.3, 0.4]
   - Weight Decay: [0.0, 1e-4]
   - Loss Type: ["mse", "smoothl1"]
   - Geometry Lambda: [0.0, 0.1] (pairwise distance regularization)
   - SA1 Radii: [[0.1,0.2,0.4], [0.05,0.1,0.2]]
   - SA2 Radii: [[0.2,0.4,0.8], [0.1,0.2,0.4]]
   - Total: 2×2×2×2×2×2×2 = 128 configurations

🔧 **Key Improvements:**
   1. **FPS Device Fix:** `farthest_point_sampling()` now checks device type - uses GPU-accelerated `furthest_point_sample` on CUDA, falls back to random sampling on CPU (prevents crash)
   2. **GPU Augmentation:** Moved data augmentation to GPU by transferring batches to device BEFORE `augment_batch()` call (rotation/scale/shift/jitter all run on GPU tensors)
   3. **Val L2 Selection:** Tracks validation L2 mean (in normalized space) per epoch; selects best config by lowest val L2 mean instead of loss (more aligned with evaluation metric)
   4. **No Val Leakage in Final Run:** Retrains best config on train+val for FINAL_EPOCHS (160) WITHOUT validation-based early stopping - saves last epoch weights to avoid leaking test info

📊 **Training Pipeline:**
   - Search phase: 80 epochs per config on train/val split (90% train → 80% train + 20% val from it)
   - Selection: picks config with lowest validation L2 mean across all landmarks
   - Final phase: retrains best config on full train+val (90%) for 160 epochs, evaluates on held-out test (10%)

💾 **Outputs:**
   - Best model: `models/pointnet2_regression_hparam_best.pth`
   - Summary JSON: `results/training_histories/hparam_search_pointnet2.json` (includes all search results, best config, test metrics)
   - Log file: `results/logs/hparam_search_pointnet2_<timestamp>.log`

🎯 **Next:** Run grid search to find optimal hyperparameters for PointNet++ architecture.

