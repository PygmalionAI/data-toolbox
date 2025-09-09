Tasks. `TrainingData` and `Task` classes for a specific dataset will be in a subfolder.


Expected structure:
```
dataset_1:
    --dataset_1.py (containing a subclass of TrainingData, to handle loading)
    --dataset_1_exampletask1.py (containing a `Task`)
    --dataset_1_exampletask2.py (containing a different `Task`)
dataset_2:
    --dataset_2.py (containing a subclass of TrainingData, to handle loading)
    --dataset_2_exampletask1.py (containing a `Task`)
    --dataset_2_exampletask2.py (containing a different `Task`)
```
And so on. I will obviously write this better when I've got stuff figured out.