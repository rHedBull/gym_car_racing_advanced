### TF logs
clear TF logs
run TF board

```bash
tensorboard --logdir logs
```

# Env

continuous there are 3 actions :

0: steering, -1 is full left, +1 is full right

1: gas
2: breaking

## Code quality

### pre-commit

This project uses [pre-commit](https://pre-commit.com/) to enforce code quality.
It must be executed manually!

```bash
pre-commit run -a
```
