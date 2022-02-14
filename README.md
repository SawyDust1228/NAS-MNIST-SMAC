# 神经网络结构搜索

## 网络结构

~~~shell
Model(
  (network): Network(
    (Conv2d_1): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
    (Relu_1): ReLU()
    (AvgPool_1): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (Linear_1): Linear(in_features=1690, out_features=100, bias=True)
    (Dropout_1): Dropout(p=0.1, inplace=False)
    (Elu_1): ELU(alpha=1.0)
    (Linear_2): Linear(in_features=100, out_features=50, bias=True)
    (Dropout_2): Dropout(p=0.1, inplace=False)
    (Tanh_1): Tanh()
    (Linear_3): Linear(in_features=50, out_features=10, bias=True)
    (Dropout_3): Dropout(p=0.1, inplace=False)
    (LogSoftMax_1): LogSoftmax(dim=None)
  )
)
~~~

## 参数说明及其范围

| 参数               | 范围           | 意义                                          |
| ------------------ | -------------- | --------------------------------------------- |
| type_pool          | 0, 1           | 0:MaxPool 1:AvgPool                           |
| type_active        | 0,1,2          | 0:Relu 1:Elu 2:Tanh                           |
| out_channels       | [5, 16]        | 通道数                                        |
| kernel_size        | 3, 5           | 卷积核的大小                                  |
| padding            | 0, 1, 2        | Padding尺寸                                   |
| linear_layer_out_1 | [100, 200]     | 线性层1的output_size                          |
| drop_out_1         | [0., 0.6]      | 线性层1的dropout                              |
| active_type_1      | 0,1,2          | 0:Relu 1:Elu 2:Tanh                           |
| linear_layer_out_2 | [20, 50]       | 线性层2的output_size                          |
| drop_out_2         | [0., 0.6]      | 线性层2的dropout                              |
| active_type_2      | 0,1,2          | 0:Relu 1:Elu 2:Tanh                           |
| linear_layer_out_3 | 10             | 线性层3的output_size                          |
| drop_out_3         | [0., 0.6]      | 线性层3的dropout                              |
| last_layer_type    | 0,1,2,3        | 0:Sigmoid 1:LogSigmoid 2:SoftMax 3:LogSoftMax |
| optim_type         | 0,1            | 0:SGD 1:Adam                                  |
| lr                 | [10e-3, 10e-1] | 学习率                                        |

## 结果

![](../../../Pictures/DeepinScreenshot_plasmashell_20220213063147.png)
