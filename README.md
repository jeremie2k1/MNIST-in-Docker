# CNN model for MNIST in Docker

Commands </br>

Build docker image
```commandline
docker build -f Dockefile -t mnist_model 
```

Check image exists
```commandline
docker images
```

Start training
```commandline
docker run mnist_model --batch_size 200 --epochs 10
```