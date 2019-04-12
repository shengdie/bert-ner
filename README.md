# bert-ner
```
python train.py --max-len 100 --epochs 5 
```
# Resplit dataset
combine the training and test data, then randomly replit with ratio [train, test, validation]
```
python train.py --max-len 100 --epochs 5 --resplit "[0.75, 0.17, 0.08]"
```
# Save weight
Save the weight with lowest loss, start form `[start-save]` epoch
```
python train.py --max-len 100 --epochs 5 --resplit "[0.75, 0.17, 0.08]" --save ./weights/pytorch_weight_test --start-save 3
```