import os
import numpy
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import *
from PIL import Image


def test_input_images(input_dir, output_dir, test_file_names):
    output_names = ['ps0-1-a-1.png', 'ps0-1-a-2.png']
    index = 0
    for name in test_file_names:
        test_file = input_dir + '/' + name
        output_file = output_dir + '/' + os.path.splitext(output_names[index])[0] + ".png"
        try:
            Image.open(test_file).save(output_file)
        except IOError:
            err = "can't convert " + test_file + " to " + output_file
            print(err)
            return False
        print("saving  to " + output_file)
        index = index + 1
    return True

def clamp(pixel_value, min, max):
    if pixel_value < min:
        pixel_value = min
    if pixel_value > max:
        pixel_value = max
    return pixel_value

def save_from_array_to_image(data_array, file_name):
    try:
        im = Image.fromarray(data_array)
        im.save(file_name)
    except IOError:
        err = "can't save " + file_name
        print(err)
        return False
    print("saving  to " + file_name)


def test_color_planes(input_dir, output_dir, test_file_names):
    test_file = input_dir + '/' + test_file_names[0]

    ##Swap the red and blue pixels of test_file
    ## output to  ps0-2-a-1.png
    output_file = output_dir + '/' + 'ps0-2-a-1.png'
    data = numpy.array(Image.open(test_file))
    copy = data.copy()
    print(data.shape, data.dtype, data.size)

    ## swap R/B in place
    data[:, :, 0] = copy[:, :, 0] + copy[:, :, 2]
    data[:, :, 2] = copy[:, :, 0] - copy[:, :, 2]
    data[:, :, 2] = copy[:, :, 0] - copy[:, :, 2]

    save_from_array_to_image(data, output_file)

    ##Create a monochrome image (img1_green) by selecting the green channel of image 1
    ##Output: ps0-2-b-1.png
    output_file = output_dir + '/' + 'ps0-2-b-1.png'
    save_from_array_to_image(copy[:, :, 1], output_file)

    ##Create a monochrome image (img1_red) by selecting the red channel of image 1
    ##Output: ps0-2-c-1.png
    output_file = output_dir + '/' + 'ps0-2-c-1.png'
    save_from_array_to_image(copy[:, :, 0], output_file)
    ## green  channels seems to preseve more details
    return True


def test_replacement_of_pixels(input_dir, output_dir):
    ##Take the inner center square region of 100x100 pixels of monochrome version of image 1 ps0-2-b-1.png
    ## and insert them into the center of monochrome version of image 2  ps0-2-c-1.png
    ## Output: Store the new image created as ps0-3-a-1.png

    image1 = output_dir + '/' + 'ps0-2-b-1.png'  ## all green
    image2 = output_dir + '/' + 'ps0-2-c-1.png'  ## all red
    output_file = output_dir + '/' + 'ps0-3-a-1.png'

    data1 = numpy.array(Image.open(image1))
    data2 = numpy.array(Image.open(image2))
    height1 = data1.shape[0]
    width1 = data1.shape[1]
    height2 = data2.shape[0]
    width2 = data2.shape[1]

    ##sanity check
    if (width1 < 100 or height1 < 100 or width2 < 100 or height2 < 100):
        print("image shape less than 100 x 100 ")
        return False
    data2[(int)(height2 / 2 - 50): (int)(height2 / 2 + 50), (int)(width2 / 2 - 50): (int)(width2 / 2 + 50)] = \
        data1[(int)(height1 / 2 - 50): (int)(height1 / 2 + 50), (int)(width1 / 2 - 50):(int)(width1 / 2 + 50)]
    save_from_array_to_image(data2, output_file)


def test_math_operation(input_dir, output_dir):
    ## What is the min and max of the pixel values of img1_green?
    ##  What is the mean? What is the standard deviation?
    ## And how did you compute these?
    image1 = output_dir + '/' + 'ps0-2-b-1.png'  ## all green
    data = numpy.array(Image.open(image1))
    copy = data.copy()
    print("min: %d " % data[:,:].min())
    print("max: %d" % data[:,:].max())
    print("standard deviation: %6f" % data[:,:].std())

    ## my way of calculating standard deviation
    sum = 0
    square_sum = 0
    min = 255
    max = 0
    x = 0
    y = 0
    height = data.shape[0]
    width = data.shape[1]
    size = width * height
    while y < height:
        x = 0
        while x < width:
            if data[y,x]< min:
                min = data[y,x]
            if data[y,x] > max:
                max = data[y,x]
            sum += data[y,x]
            square_sum += (int(data[y,x]) * int(data[y,x]))

            x += 1
        y += 1
    mean = sum / size
    deviation = numpy.sqrt(square_sum/ size - mean * mean)
    print("min: %d " % min)
    print("max: %d" % max)
    print("mean: %6f" % mean)
    print("standard deviation: %6f" % deviation)

    ##Subtract the mean from all pixels, then divide by standard deviation, then multiply by 10
    ## (if your image is 0 to 255) or by 0.05 (if your image ranges from 0.0 to 1.0).
    ## Now add the mean back in
    ##Output: ps0-4-b-1.png
    x = y = 0
    while y < height:
        x = 0
        while x < width:
            temp = float(data[ y, x])
            temp = ((temp - mean ) / deviation ) * 10 + mean
            data[y, x] = np.uint8(clamp(temp, 0, 255))
            x += 1
        y += 1

    output_file = output_dir + '/' + 'ps0-4-b-1.png'
    save_from_array_to_image(data,output_file)

    ##Shift img1_green to the left by 2 pixels.
    ## Output: ps0-4-c-1.png
    data = copy.copy()
    y = 0
    while y < height:
        x = 2
        while x < width:
            data[y, x-2] = data[y, x]
            x += 1
        y += 1
    data[:, width-2:width-1 ] = 0
    output_file = output_dir + '/' + 'ps0-4-c-1.png'
    save_from_array_to_image(data, output_file)


    ##Subtract the shifted version of img1_green from the original, and save the difference image.
    ## Output: ps0-4-d-1.png (make sure that the values are legal when you write the image
    ##so that you can see all relative differences), text response: What do negative pixel values mean anyways?
    x = y = 0
    while y < height:
        x = 0
        while x < width:
            copy[y, x] = np.uint8(clamp(int(copy[y, x]) - int(data[y, x]), 0, 255 ))
            x += 1
        y += 1
    ##copy[:, :] -= data[:, :]        ## this will generate negative pixels and looks wash out
    output_file = output_dir + '/' + 'ps0-4-d-1.png'
    save_from_array_to_image(copy, output_file)

def test_noises(input_dir,output_dir):
    image1 = input_dir + '/' + input_file_names[0]  ## original
    data = numpy.array(Image.open(image1))

    height = data.shape[0]
    width = data.shape[1]
    size = data.shape[0] * data.shape[1]
    ##Take the original colored image (image 1) and start adding Gaussian noise to the pixels in the green channel.
    ##Increase sigma until the noise is somewhat visible.
    ##Output: ps0-5-a-1.png, text response: What is the value of sigma you had to use?
    mean = data[:,:].mean()
    sigma_range = [1, int(mean/64), int(mean/32), int(mean/16), int(mean/8), int(mean/4)]
    for sigma in sigma_range:
        copy = data.copy()
        #(row, col, ch) = data.shape()
        noise = numpy.random.normal(0, sigma, size)
        noise = noise.reshape((height,width))
        copy[:,:,1] = copy[:,:,1] + noise
        output_file = output_dir + '/' + 'ps0-5-a-1' + '_sigma_' + str(sigma) +'.png'
        save_from_array_to_image(copy, output_file)

    ##looks like noise starts to be visible around sigma = 16 which is 1/8 of mean value
    ## plot -- how to plot them all together
    for sigma in sigma_range:
        output_file = output_dir + '/' + 'ps0-5-a-1' + '_sigma_' + str(sigma) + '.png'
        img = array(Image.open(output_file))
        imshow(img)
        title('Plotting:')
        show()

    ##Now, instead add that amount of noise to the blue channel.
    ##Output: ps0-5-b-1.png
    ##Which looks better? Why?
    ##Output: Text response
    for sigma in sigma_range:
        copy = data.copy()
        #(row, col, ch) = data.shape()
        noise = numpy.random.normal(0, sigma, size)
        noise = noise.reshape((height,width))
        copy[:,:,2] = copy[:,:,2] + noise
        output_file = output_dir + '/' + 'ps0-5-b-1' + '_sigma_' + str(sigma) +'.png'
        save_from_array_to_image(copy, output_file)

    ##looks like human eyes to noise on blue channel is less sensitive

if __name__ == "__main__":
    import argparse

    input_dir = "./input"
    output_dir = "./output"
    input_file_names = ['Mandrill_512x512.tiff', 'Sailboat_on_lake_512x512.tiff']
    test_items = ['input_images', 'color_planes', 'replacement_of_pixels', 'math_operation', 'noises']
    parser = argparse.ArgumentParser(description="PS0")
    parser.add_argument("-test", "--test", dest="test_item",
                        help="test item(input_images/color_planes/replacement_of_pixels/math_operation/noises/all)",
                        action="store")

    args = parser.parse_args()
    test_item = args.test_item
    print("testing " + test_item)

    if test_item == "input_images":
        test_input_images(input_dir, output_dir, input_file_names)
    elif test_item == "color_planes":
        test_color_planes(input_dir, output_dir, input_file_names)
    elif test_item == "replacement_of_pixels":
        test_replacement_of_pixels(input_dir, output_dir)
    elif test_item == "math_operation":
        test_math_operation(input_dir, output_dir)
    elif test_item == "noises":
        test_noises(input_dir, output_dir)
    elif test_item == "all":
        test_input_images(input_dir, output_dir, input_file_names)
        test_color_planes(input_dir, output_dir, input_file_names)
        test_replacement_of_pixels(input_dir, output_dir)
        test_math_operation(input_dir, output_dir)
        test_math_operation(input_dir, output_dir)
        test_noises(input_dir, output_dir)
    else:
        print("unrecoginzied test")

        ##print("return " + str(result))
