model=model005-resnet34
gpu=2
fold=4
conf=./conf/${model}.py

python -m src.cnn.main train ${conf} --debug --fold ${fold} --gpu ${gpu}
