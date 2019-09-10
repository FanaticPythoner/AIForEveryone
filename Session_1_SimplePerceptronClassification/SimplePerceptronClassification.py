#Copyright Nathan Trudeau Informatique, 2019, https://ntinformatique.ca/en/services/private-courses.html. Github: https://github.com/FanaticPythoner

#Numpy is a must-know in Python: If you do math, this library is your best friend.
import numpy

#This is the library we will use to draw the Canvas.
from tkinter import Canvas, Tk

#We need to generate random positions for the dots.
from random import randint

#Because our Perceptron will be damn fast, we need to slow it down a bit
#in order to be able to see how he is training.
from time import sleep


class PerceptronCanvas:
    '''This is the Canvas class which will display how our Perceptron is doing.'''

    def __init__(self, dimensionX, dimensionY):
        #Setting the dimensions of the Canvas.
        self.dimensionX = dimensionX
        self.dimensionY = dimensionY
        self.master = Tk()
        self.master.geometry(str(self.dimensionX) + 'x' + str(self.dimensionY))
        self.main = Canvas(self.master, width=self.dimensionX, height=self.dimensionY)
        self.main.pack()

        #Creating the line which will sperate the Canvas in two.
        self.main.create_line(self.dimensionX, self.dimensionY, 0, 0, fill="red")

        #From the linear equation y=ax+b, I need to know "a" and "b" to calculate the expected result
        #based on coordonate for when we train our Perceptron.
        self.a = (0 - self.dimensionY)/(0 -self.dimensionX )
        self.b = self.a * self.dimensionX - self.dimensionY

        #These are the dictionnaries which will hold the data of the dots we will
        #put randomly on the Canvas.
        self.pointObjects = {}
        self.pointPositions = {}

    def createDotsAndTraningData(self, numberOfDots):
        '''This function will create random dots on the Canvas and put in our dictionnaries
           the dots' data (x, y, expected prediction of -1 or 1).'''

        for number in range(numberOfDots):
            #Generating random X and Y for the dot.
            x1 = randint(0,self.dimensionX - 20)
            y1 = randint(0,self.dimensionY - 20)

            #Creating the dot, putting it on the Canvas and moving it to the random
            #X and Y we just generated.
            dot = self.main.create_oval(0,0,10,10)
            self.main.move(dot, x1, y1)

            #Adding the dot object into one of the dictionnary.
            self.pointObjects['dot' + str(number)] = dot

            #Here, this is why i previously calculated "a" and "b" from the y=ax+b formula:
            #I need them to know if the dot is below or over the line that separate the Canvas
            #in two.
            OneOrMinusOne = 0
            if y1 < (self.a * x1) + self.b:
                OneOrMinusOne = 1
            elif y1 > (self.a * x1) + self.b:
                OneOrMinusOne = -1
            else:
                OneOrMinusOne = 1

            #For each dot we put on the Canvas we need to refresh it in order to
            #see the changes we just made.
            self.master.update_idletasks()
            self.master.update()

            #Adding the X, Y and expected prediction result into the second dictionnary.
            #This will be used for the training process.
            self.pointPositions['dot' + str(number)] = (x1, y1, OneOrMinusOne)

class Perceptron:
    def __init__(self, perceptronCanvas):
        #Weights 1 and 2, along with the Learning Rate and the Bias.
        self.W1 = 0
        self.W2 = 0
        self.learning_rate = 0.01
        self.B1 = 0

        #This variable is our Perceptron's Canvas. We will need to it access the data of
        #the random dots we generated earlier.
        self.perceptronCanvas = perceptronCanvas

    def predict(self, X1, X2):
        # Initialize the Inputs matrix and Weights matrix.
        inputsMatrix = numpy.array([X1,X2])
        weightsMatrix = numpy.array([self.W1,self.W2])

        # This is our matrix dot product of the Weights and Inputs, plus the Bias.
        V = numpy.average(numpy.dot(inputsMatrix, weightsMatrix) + self.B1)
        
        # Now we simply return the result of our previous value passed into the Sign function.
        return numpy.sign(V)

    def trainFromDataset(self, iter_count=2000,sleep_In_Second_Between_Each_Itteration=0.01):
        '''This is where all the magic happen: This method will train our Perceptron to recognize if either the dot is
           above or under the line we drew earlier on our Canvas.'''

        #This is our training data that we previously stored in a dictionnary earlier when 
        #creating the dots on the canvas.
        dataset = self.perceptronCanvas.pointPositions

        #Each time our Perceptron is able to predict correctly if a dot is below or above the line,
        #we will add the dot data into this dictionnary to not overdo it.
        for _ in range(iter_count):
            i = randint(0,len(dataset) - 1)
            x, y, answer = dataset.get('dot' + str(i))
            currentPoint = self.perceptronCanvas.pointObjects.get('dot' + str(i))
            prediction = self.predict(x,y)

            #Calculate the error and deltas of the weights
            error = answer - prediction
            delta_W1 = error * x
            delta_W2 = error * y

            #Calculate the new Weights and the new Bias
            self.W1 += delta_W1 * self.learning_rate
            self.W2 += delta_W2 * self.learning_rate
            self.B1 += error * self.learning_rate

            if error != 0:
                if answer == -1:
                    self.perceptronCanvas.main.itemconfig(currentPoint,fill="green")
                    self.perceptronCanvas.master.update_idletasks()
                    self.perceptronCanvas.master.update()
                else:
                    self.perceptronCanvas.main.itemconfig(currentPoint,fill="red")
                    self.perceptronCanvas.master.update_idletasks()
                    self.perceptronCanvas.master.update()
            else:
                if prediction == -1:
                    self.perceptronCanvas.main.itemconfig(currentPoint,fill="red")
                    self.perceptronCanvas.master.update_idletasks()
                    self.perceptronCanvas.master.update()
                else:
                    self.perceptronCanvas.main.itemconfig(currentPoint,fill="green")
                    self.perceptronCanvas.master.update_idletasks()
                    self.perceptronCanvas.master.update()

            sleep(sleep_In_Second_Between_Each_Itteration)

    
#We initialize a Canvas of 800 pixels by 800 pixels.
pCanvas = PerceptronCanvas(800,800)

#We generate 50 random dots on our Canvas.
pCanvas.createDotsAndTraningData(50)

#We create our awesome Perceptron, passing our previously created Canvas as an argument.
neuralNetwork = Perceptron(pCanvas)

#We train it and watch the magic happen.
neuralNetwork.trainFromDataset(iter_count=2000,
                               sleep_In_Second_Between_Each_Itteration=0.1)
