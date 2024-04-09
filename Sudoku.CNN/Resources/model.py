from keras import Input, Model, Sequential
from keras.layers import Add, Conv2D, MaxPooling2D, Concatenate, Activation, Dropout, Flatten, Dense, Reshape, BatchNormalization
from keras.regularizers import l2


def get_model():

    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))

    return model

def get_complete_model():
    input_layer = Input(shape=(9, 9, 1))

    # Base convolutional layers
    conv_base1 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
    conv_base1 = BatchNormalization()(conv_base1)
    conv_base2 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(conv_base1)
    conv_base2 = BatchNormalization()(conv_base2)
    conv_base3 = Conv2D(256, kernel_size=(1,1), activation='relu', padding='same')(conv_base2)
    conv_base3 = BatchNormalization()(conv_base3)
    
    # Specialized convolutional layers for Sudoku rules, each followed by BatchNormalization
    # 3x3 convolutions with 3x3 strides for boxes
    conv_boxes = Conv2D(128, (3, 3), strides=(3, 3), activation='relu', padding='same')(input_layer)
    conv_boxes = BatchNormalization()(conv_boxes)

    # 1x9 convolutions for rows
    conv_rows = Conv2D(128, (1, 9), strides=(1, 1), activation='relu', padding='valid')(input_layer)
    conv_rows = BatchNormalization()(conv_rows)

    # 9x1 convolutions for columns
    conv_columns = Conv2D(128, (9, 1), strides=(1, 1), activation='relu', padding='valid')(input_layer)
    conv_columns = BatchNormalization()(conv_columns)

    # Flatten and concatenate features from all paths
    flattened_base = Flatten()(conv_base3)
    flattened_boxes = Flatten()(conv_boxes)
    flattened_rows = Flatten()(conv_rows)
    flattened_columns = Flatten()(conv_columns)
    concatenated = Concatenate()([flattened_base, flattened_boxes, flattened_rows, flattened_columns])

    # Dense layers after feature concatenation
    dense_layer = Dense(81*9, activation='relu', kernel_regularizer=l2(1e-4))(concatenated)
    dense_layer = BatchNormalization()(dense_layer)  # Adding BatchNormalization here as well
    output_layer = Reshape((9, 9, 9))(dense_layer)
    output_layer = Activation('softmax')(output_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
