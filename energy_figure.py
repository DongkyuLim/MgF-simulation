from cProfile import label
from multiprocessing.sharedctypes import Value
from re import L
from turtle import color
import matplotlib.pyplot as plt
import numpy as np

class energy_ket:
    def __init__(self,energy,label:str,xmin=0,xmax=1,**kwargs):
        self.energy = energy
        self.label = label
        self.xmin = xmin
        self.xmax = xmax
        self.xstep = 1
        self.kwargs=kwargs

    def changeX(self,xmin,xmax):
        self.xmin = xmin
        self.xmax = xmax
    
    def changeStep(self,steps):
        self.xstep = steps

class transmission_line:
    def __init__(self,ket1:energy_ket,ket2:energy_ket,up_offset=0,down_offset=0,arrow_type=False,color="C0",**props):
        if ket1.xstep != ket2.xstep:
            raise AttributeError("I`m sorry, we can only draw vertical line.")
        
        if ket1.energy>ket2.energy:
            up_ket=ket1
            down_ket=ket2
        else:
            up_ket=ket2
            down_ket=ket1
        
        self.xmin=up_ket.xmin
        self.xmax=up_ket.xmax
        self.xstep = up_ket.xstep
        self.up_energy = up_ket.energy+up_offset
        self.down_energy = down_ket.energy+down_offset
        self.arrow_type=arrow_type
        self.color = color
        self.property = props
        if not props:
            self.property = {"props" : {"facecolor" : "blue","edgecolor" : "blue"}}


class split_line:
    def __init__(self,ket1:energy_ket,ket2:energy_ket,color="C0",**props):
        if ket1.xmax < ket2.xmin:
            self.leftdata = [ket1.xmax,ket2.xmin]
            self.rightdata= [ket1.energy,ket2.energy]
        elif ket2.xmax < ket1.xmin:
            self.leftdata = [ket2.xmax,ket1.xmin]
            self.rightdata= [ket2.energy,ket1.energy]
        self.color = color
        self.props = props

class energy_diagram:
    def __init__(self,step_size=1.):
        self.Xoffset = 0
        self.Yoffset = 0
        self.labels = dict()
        self.labelname = list()
        self.Xstep = 1
        self.Xpos = dict()
        self.Xpos[1] = [0,step_size]
        self.__transmission_label={1 : list()}
        self.splits = list()

    def add_energy(self,energy,label:str,**props):
        if label in self.labelname:
            raise ValueError("There is same name in labels")
        ket = energy_ket(energy+self.Yoffset,label,**props)
        ket.changeX(self.Xpos[self.Xstep][0],self.Xpos[self.Xstep][1])
        ket.changeStep(self.Xstep)
        self.labels[label]=ket

    
    def next_X(self,add_offset = 2,step_size=1.,reset_Y=True):
        if add_offset<1:
            raise ValueError("Offset should be right side than previous step.")
        self.Xoffset+=add_offset
        self.Xstep+=1
        self.Xpos[self.Xstep] = [self.Xoffset,step_size + self.Xoffset]
        self.__transmission_label[self.Xstep] = list()
        if reset_Y:
            self.Yoffset = 0
        

    def next_Y(self,add_offset = 300,reset_X=True):
        self.Yoffset+=add_offset
        if reset_X:
            self.Xoffset=0

    def transmission(self,label1,label2,up_offset=0.,down_offset=0.,arrow_type=False,**kwargs):
        if not label1 in self.labels or not label2 in self.labels:
            raise ValueError("There is no such label.")
        ket1 = self.labels[label1]
        ket2 = self.labels[label2]
        self.__transmission_label[ket1.xstep].append(transmission_line(ket1=ket1,ket2=ket2,up_offset=up_offset,down_offset=down_offset,arrow_type=arrow_type,props=kwargs))

    def split(self,label1,label2,**kwargs):
        ket1 = self.labels[label1]
        ket2 = self.labels[label2]
        self.splits.append(split_line(ket1,ket2,props=kwargs))

    def plot(self,axis_on=True,saving=False,name=None):
        fig, ax = plt.subplots(1,1,figsize=(60,45))

        ax.set_title(name)
        for ket in self.labels:
            ax.hlines(self.labels[ket].energy,self.labels[ket].xmin,self.labels[ket].xmax,"black",**self.labels[ket].kwargs)
        

        for step_num in self.__transmission_label:
            counter = 0
            tot = len(self.__transmission_label[step_num])
            length = abs(self.Xpos[step_num][1]-self.Xpos[step_num][0])
            for temp in self.__transmission_label[step_num]:
                ticks = length/(tot+1)*(counter+1)
                if temp.arrow_type:
                    ax.annotate("",xy=(temp.xmin+ticks,temp.up_energy),xytext=(temp.xmin+ticks,temp.down_energy),arrowprops=dict(facecolor=temp.color, edgecolor=temp.color,**temp.property['props']))
                else:
                    ax.vlines(temp.xmin+ticks,temp.down_energy,temp.up_energy)
                counter+=1

        for spline in self.splits:
            ax.plot(spline.leftdata,spline.rightdata,color=spline.color,**spline.props["props"])

        if not axis_on:
            plt.axis("off")
        if saving:
            if not name:
                raise ValueError("Enter the name.")
            fig.savefig(name + ".png")


        plt.show(block=True)



if __name__=="__main__":
    diag = energy_diagram()

    diag.add_energy(0,"Xstate")
    diag.next_Y(300)
    diag.add_energy(0,"Astate")
    diag.next_X()
    diag.add_energy(0,"F=0")
    diag.add_energy(110,"F=1+")
    diag.add_energy(120,"F=2")
    diag.add_energy(-120,"F=1-")

    diag.next_Y(300)
    diag.add_energy(0,"F`=0")
    diag.add_energy(10,"F`=1")

    diag.transmission("F=1-", "F`=1",arrow_type=1,width=0.5,headwidth=5,headlength=10)
    diag.transmission("F=0","F`=1",arrow_type=1,width=0.5,headwidth=5,headlength=10)
    diag.transmission("F=1+","F`=1",arrow_type=1,width=0.5,headwidth=5,headlength=10)
    diag.transmission("F=2","F`=1",arrow_type=1,width=0.5,headwidth=5,headlength=10)

    diag.split("Xstate","F=0")
    diag.split("Xstate","F=1-")
    diag.split("Xstate","F=2")
    diag.split("Xstate","F=1+")
    diag.split("Astate","F`=0")
    diag.split("Astate","F`=1")
    diag.plot(name="MgF energy diagram")


