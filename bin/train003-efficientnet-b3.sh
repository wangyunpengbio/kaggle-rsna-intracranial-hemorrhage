model=model003-efficientnet-b3
gpu=0
fold=2
conf=./conf/${model}.py

python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
