import kfp
from kfp import dsl

# Preprocessing operation
def preprocess_op():
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='altair4357/kfp:preprocess',  # Image to be used
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'y_train': '/app/y_train.npy',
            'x_test': '/app/x_test.npy',
            'y_test': '/app/y_test.npy',
        }
    )

# Training operation
def train_op(x_train, y_train, x_test, y_test):
    return dsl.ContainerOp(
        name='Train Model',
        image='altair4357/kfp:train',  # Image to be used
        arguments=[
            '--x_train', x_train,  # Input arguments
            '--y_train', y_train,
            '--x_test', x_test,
            '--y_test', y_test
        ],
        file_outputs={
            'model': '/app/model.h5'  # Output file
        }
    )

# Evaluation operation
def evaluate_op(x_test, y_test, model):
    return dsl.ContainerOp(
        name='Evaluate Model',
        image='altair4357/kfp:evaluate',  # Image to be used
        arguments=[
            '--x_test', x_test,  # Input arguments
            '--y_test', y_test,
            '--model', model
        ]
    )

# Define the pipeline
@dsl.pipeline(
   name='MNIST Digit Recognition Pipeline',
   description='A pipeline that trains and tests a model on the MNIST dataset.'
)
def mnist_pipeline():
    _preprocess_op = preprocess_op()
    
    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test'])
    ).after(_preprocess_op)

    _evaluate_op = evaluate_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']), # Typo fixed here
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)

# Main function to compile the pipeline
if __name__ == "__main__":
    import kfp.compiler as compiler
    import os
    compiler.Compiler().compile(mnist_pipeline, os.path.expanduser("~") + "/mnist-pipeline.tar.gz")
