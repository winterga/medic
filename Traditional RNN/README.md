# Traditional RNN

## How to Run

Since the code is modulated using `__init.__py`, we need to use the correct command to execute the code.

Although it is not required, we use `argparse` to break up the options available, such as `--base_dir`.

### For Jane and Greyson

You need to make sure that you have a `src` folder inside your code.

You need to make sure you have a `__init.py__` inside `src`.

From here, you can use the modules option to run the code:

```
python -m src.<PYTHON_FILENAME>
```

So, the way I have mine, I do:

```
python -m src.run_experiment --base_dir /home/user/Documents/GitHub/medic/data/images_3/
```