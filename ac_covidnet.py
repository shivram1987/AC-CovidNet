from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, RepeatVector, Lambda, Multiply, Conv2D, Reshape
from tensorflow.keras.layers import BatchNormalization, Concatenate, AveragePooling2D, Flatten, Conv2DTranspose, Average, add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, MaxPooling2D, DepthwiseConv2D
from attention import *

def Pepx(filters, kernel=(1, 1)):
    def inside(x):
        x = Conv2D(filters, kernel, padding='same', activation = 'relu') (x)
        x = Conv2D(filters, kernel, padding='same', activation = 'relu') (x)
        x = DepthwiseConv2D((3, 3), padding='same') (x)
        x = Conv2D(filters, kernel, padding='same', activation = 'relu') (x)
        x = Conv2D(filters, kernel, padding='same', activation = 'relu') (x)
        return x
    return inside

# usage:
# x = Pepx(params) (x)

def ACCovidNet():
  i = Input(shape=(480, 480, 3))
  ip = MaxPooling2D(pool_size=(2, 2), padding='same') (i)
  c1 = Conv2D(kernel_size=(7, 7), filters=56, activation='relu', padding='same') (ip)
  c1p = MaxPooling2D(pool_size=(2, 2), padding='same') (c1)

  ###### PEPX BLOCKS + Conv1x1 ######
  p1_1 = Pepx(56) (c1p)
  cr1 = Conv2D(kernel_size=(1, 1), filters=56, activation='relu', padding='same') (c1p)
  concat_cr1_p12 = add([p1_1, cr1])
  p1_2 = Pepx(56) (concat_cr1_p12)
  concat_cr1_p13 = add([p1_2, cr1, p1_1])
  p1_3 = Pepx(56) (concat_cr1_p13)
  concat_p1_cr2 = add([p1_1, p1_2, p1_3, cr1])
  cr1p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_p1_cr2)

  a1 = attention_block_2d(p1_3, Average()([p1_1, p1_2]))
  concat_cr1_p21 = add([cr1, a1])
  p1_3p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_cr1_p21)

  p2_1 = Pepx(112) (p1_3p)
  cr2 = Conv2D(kernel_size=(1, 1), filters=112, activation='relu', padding='same') (cr1p)
  concat_cr2_p22 = add([p2_1, cr2])
  p2_2 = Pepx(112) (concat_cr2_p22)
  concat_cr2_p23 = add([p2_2, cr2, p2_1])
  p2_3 = Pepx(112) (concat_cr2_p23)
  concat_cr2_p24 = add([p2_3, cr2, p2_1, p2_2])
  p2_4 = Pepx(112) (concat_cr2_p24)
  concat_p2_cr3 = add([p2_1, p2_2, p2_3, p2_4, cr2])
  cr2p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_p2_cr3)

  a2 = attention_block_2d(p2_4, Average()([p2_1, p2_2, p2_3]))
  concat_cr2_p31 = add([cr2, a2])
  p2_4p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_cr2_p31)

  p3_1 = Pepx(216) (p2_4p)
  cr3 = Conv2D(kernel_size=(1, 1), filters=216, activation='relu', padding='same') (cr2p)
  concat_cr3_p32 = add([p3_1, cr3])
  p3_2 = Pepx(216) (concat_cr3_p32)
  concat_cr3_p33 = add([p3_2, cr3, p3_1])
  p3_3 = Pepx(216) (concat_cr3_p33)
  concat_cr3_p34 = add([p3_3, cr3, p3_1, p3_2])
  p3_4 = Pepx(216) (concat_cr3_p34)
  concat_cr3_p35 = add([p3_4, cr3, p3_1, p3_2, p3_3])
  p3_5 = Pepx(216) (concat_cr3_p35)
  concat_cr3_p36 = add([p3_5, cr3, p3_1, p3_2, p3_3, p3_4])
  p3_6 = Pepx(216) (concat_cr3_p36)
  concat_p3_cr4 = add([p3_1, p3_2, p3_3, p3_4, p3_5, p3_6, cr3])
  cr3p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_p3_cr4)

  a3 = attention_block_2d(p3_6, Average()([p3_1, p3_2, p3_3, p3_4, p3_5]))
  concat_cr3_p41 = add([cr3, a3])
  p3_6p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_cr3_p41)

  p4_1 = Pepx(424) (p3_6p)
  cr4 = Conv2D(kernel_size=(1, 1), filters=424, activation='relu', padding='same') (cr3p)
  concat_cr4_p4_2 = add([p4_1, cr4])
  p4_2 = Pepx(424) (concat_cr4_p4_2)
  concat_cr4_p4_3 = add([p4_2, cr4, p4_1])
  p4_3 = Pepx(424) (p4_2)
  #########################################

  af = attention_block_2d(cr4, Average()([p4_1, p4_2, p4_3]))
  f = Flatten() (af)
  fc1 = Dense(1024, activation = 'relu')(f)
  fc2 = Dense(256, activation = 'relu')(fc1)

  fc3 = Dense(3, activation='softmax')(fc2)

  model = Model(inputs = i, outputs = fc3)
  return model

if __name__ == '__main__':
    model = ACCovidNet()
    model.summary()
