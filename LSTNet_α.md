### 将LSTNet与α预测结合

因为α预测的是确诊总数，而之前LSTNet预测的是新增的死亡，治愈和确诊人数，因此现在将LSTNet的输入数据改为确诊人数

#### 1. 使用α对浙江省，湖北省和非湖北区域总确诊人数进行预测

|         | 浙江省 | 湖北省  | 非湖北区域 |
| :-----: | :----: | :-----: | :--------: |
| MSEloss | 56586  | 3733102 |   75565    |

#### 2. 使用LSTNet对三种数据对确诊人数进行预测

|            | 浙江省 |  湖北省   | 非湖北区域 |
| :--------: | :----: | :-------: | :--------: |
| train-loss |  488   |   11952   |    4660    |
| test-loss  | 237595 | 820394422 |  28836276  |

浙江省的数据有缺陷，因为浙江省后几天的确诊人数没有变化，而test使用的是位于后面的数据

#### 3. 将LSTNet与α结合进行预测

模型的输入为7天的数据

方法一：模型中的α由输入的7天的数据的α值的中位数决定，而不参与模型训练。这意味着每当输入不同的x时，α的值是不同的

|            | 浙江省 | 湖北省  | 非湖北区域 |
| :--------: | :----: | :-----: | :--------: |
| train-loss |   41   |   743   |     94     |
| test-loss  |  634   | 2138477 |   170959   |

方法二：模型中的α由模型训练得出。这意味着对于任何的输入x，α的值都是一样的。**因为还没了解清楚如何对α进行梯度下降，所以还没实现**