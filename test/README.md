# Docker container

## Build the container

move to the directory where the Dockerfile is located and run the following command

```bash
docker build -t image-name .
```

## Run the container locally

```bash
docker run -e WANDB_API_KEY=WANDB_API_KEY  image-name
```

## Deploy

For now the build and tested container is pushed to AWS ECR. It can then be pulled and run on vast.ai or AWS sagemaker.
We are using the awscli to push to AWS. Therefor we must include the following environment variables in the docker run command:

### push docke image to AWS ECR

Tag your local Docker image to match the ECR repository.
```bash
docker tag mock-docker-test:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/mock-docker-test:latest
```

Push the tagged image to your ECR repository.
```bash
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/mock-docker-test:latest
```

Also follow the guide inthe AWS ECR console to get the docker login command.

The full run command is then:
```bash
docker run -e WANDB_API_KEY=WANDB_API -e AWS_ACCESS_KEY_ID=MY_ACCESS_KEY -e AWS_SECRET_ACCESS_KEY=MY_SECRET_KEY -e S3_PATH=S3_URL image-name
```

### AWS Sagemaker


### Vast.ai

Requirements:
- vast.ai account
- wand API key
- AWS S3 bucket


# TODO
* [ ] find setup for smaller image size
* [ ] find container setup that works for vast.ai and AWS sagemaker
