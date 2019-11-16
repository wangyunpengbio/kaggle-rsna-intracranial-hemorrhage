model=model002-efficientnet-b3
gpu=3
fold=1
conf=./conf/${model}.py

python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
