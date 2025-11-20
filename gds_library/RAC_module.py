# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:51:00 2021

@author: kist
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import itertools
from matplotlib import pyplot as plt
from numpy import sqrt, pi, cos, sin, log, exp, sinh,tan
from scipy.special import iv as besseli
from scipy.optimize import fmin, fminbound
from scipy import integrate
import phidl
from phidl import Device, Layer, LayerSet, make_device, Path, CrossSection, Group
from phidl import quickplot as qp # Rename "quickplot()" to the easier "qp()"
import phidl.geometry as pg
import phidl.routing as pr
import phidl.utilities as pu
import phidl.path as pp
import pandas as pd

"""
Created on Thu May 14 15:32:48 2020
"<<pg.ellipse()"
@author: User01
"""
def waveguide(width = 1, length = 10, layer=0):
    WG = Device('waveguide')
    WG.add_polygon( [(0, width/2), (length, width/2), (length, -width/2), (0, -width/2)], layer=layer )
    WG.add_port(name = 'wgport1', midpoint = [0,0], width = width, orientation = 180)
    WG.add_port(name = 'wgport2', midpoint = [length,0], width = width, orientation = 0)
    return WG


def Elec_waveguide(width = 1,length = 10, E_gap=5,E_width=10, E_pad_leg=190,E_padx=100,E_pady=100, layer=0):
    EWG = Device('Elec_waveguide')
    EWG.add_polygon( [(0, width/2), (length, width/2), (length, -width/2), (0, -width/2)], layer=layer )
    EWG.add_polygon( [(0, E_width+E_gap/2), (length, E_width+E_gap/2), (length, E_gap/2), (0, E_gap/2)], layer=layer+1 )
    EWG.add_polygon( [(0,-E_width-E_gap/2), (length,-E_width-E_gap/2), (length,-E_gap/2), (0,-E_gap/2)], layer=layer+1 )

    EWG.add_polygon( [(E_width/2-E_padx/2,E_width+E_gap/2+E_pad_leg ), (E_width/2+E_padx/2, E_width+E_gap/2+E_pad_leg), (E_width/2+E_padx/2, E_pady+E_width+E_gap/2+E_pad_leg), (E_width/2-E_padx/2, E_pady+E_width+E_gap/2+E_pad_leg),], layer=layer+1 )
    EWG.add_polygon( [(E_width/2-E_padx/2,-E_width-E_gap/2-E_pad_leg ), (E_width/2+E_padx/2, -E_width-E_gap/2-E_pad_leg), (E_width/2+E_padx/2, -E_pady-E_width-E_gap/2-E_pad_leg), (E_width/2-E_padx/2, -E_pady-E_width-E_gap/2-E_pad_leg),], layer=layer+1 )
    EWG.add_polygon( [(E_width/2-E_width/2,E_width+E_gap/2 ), (E_width/2+E_width/2, E_width+E_gap/2), (E_width/2+E_width/2, E_pad_leg+E_width+E_gap/2), (E_width/2-E_width/2, E_pad_leg+E_width+E_gap/2),], layer=layer+1 )
    EWG.add_polygon( [(E_width/2-E_width/2,-E_width-E_gap/2 ), (E_width/2+E_width/2, -E_width-E_gap/2), (E_width/2+E_width/2, -E_pad_leg-E_width-E_gap/2), (E_width/2-E_width/2, -E_pad_leg-E_width-E_gap/2),], layer=layer+1 )
 
    
    EWG.add_polygon( [(length-E_width/2-E_padx/2,E_width+E_gap/2+E_pad_leg ), (length-E_width/2+E_padx/2, E_width+E_gap/2+E_pad_leg), (length-E_width/2+E_padx/2, E_pady+E_width+E_gap/2+E_pad_leg), (length-E_width/2-E_padx/2, E_pady+E_width+E_gap/2+E_pad_leg),], layer=layer+1 )
    EWG.add_polygon( [(length-E_width/2-E_padx/2,-E_width-E_gap/2-E_pad_leg ), (length-E_width/2+E_padx/2, -E_width-E_gap/2-E_pad_leg), (length-E_width/2+E_padx/2, -E_pady-E_width-E_gap/2-E_pad_leg), (length-E_width/2-E_padx/2, -E_pady-E_width-E_gap/2-E_pad_leg),], layer=layer+1 )
    EWG.add_polygon( [(length-E_width/2-E_width/2,E_width+E_gap/2 ), (length-E_width/2+E_width/2, E_width+E_gap/2), (length-E_width/2+E_width/2, E_width+E_gap/2+E_pad_leg), (length-E_width/2-E_width/2, +E_width+E_gap/2+E_pad_leg),], layer=layer+1 )
    EWG.add_polygon( [(length-E_width/2-E_width/2,-E_width-E_gap/2 ), (length-E_width/2+E_width/2, -E_width-E_gap/2), (length-E_width/2+E_width/2, -E_width-E_gap/2-E_pad_leg), (length-E_width/2-E_width/2, -E_width-E_gap/2-E_pad_leg),], layer=layer+1 )

    
    EWG.add_port(name = 'wgport1', midpoint = [0,0], width = width, orientation = 180)
    EWG.add_port(name = 'wgport2', midpoint = [length,0], width = width, orientation = 0)
    return EWG

def taper(WGwidth = 1, min_width=0.5, taper_length = 3,shift=0.5, layer = 0):
    """ Creates an taper"""
    t = np.linspace(0, taper_length, 2)
    upper_points_x = (t).tolist()
    upper_points_y = (WGwidth/2-(WGwidth/2-min_width/2-shift/2)/taper_length*t).tolist()
    lower_points_x = (t).tolist()
    lower_points_y = (-WGwidth/2+(WGwidth/2-min_width/2+shift/2)/taper_length*t).tolist()
    xpts = upper_points_x + lower_points_x[::-1]
    ypts = upper_points_y + lower_points_y[::-1]

    D = Device('taper')
    D.add_polygon(points = (xpts,ypts), layer = layer)    
    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [taper_length,shift/2], width = min_width, orientation = 0)
    D.info['length'] = taper_length
    return D

def taper_port(WGwidth = 1, WGlength=5,  taper_width=0.5, taper_length = 3,buffer_length=1, layer = 0):
    """ Creates an taper"""
    D = Device('waveguide')
    D.add_polygon( [(0, WGwidth/2), (WGlength, WGwidth/2), (WGlength+taper_length, taper_width/2), (WGlength+taper_length+buffer_length, taper_width/2), (WGlength+taper_length+buffer_length, -taper_width/2), (WGlength+taper_length, -taper_width/2), (WGlength, -WGwidth/2), (0, -WGwidth/2)] ,layer=layer)
    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [WGlength+taper_length+buffer_length,0], width = taper_width, orientation = 0)
    D.info['length'] = taper_length
    return D

def taper_port2(WGleft = 1, WGleftlength=5,  WGright=0.5,WGrightlength=1, taper_length = 3, layer = 0):
    """ Creates an taper"""
    D = Device('waveguide')
    D.add_polygon( [(0, WGleft/2), (WGleftlength, WGleft/2), (WGleftlength+taper_length, WGright/2), (WGleftlength+taper_length+WGrightlength, WGright/2), (WGleftlength+taper_length+WGrightlength, -WGright/2), (WGleftlength+taper_length, -WGright/2), (WGleftlength, -WGleft/2), (0, -WGleft/2)] ,layer=layer)
    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGleft, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [WGleftlength+taper_length+WGrightlength,0], width = WGright, orientation = 0)
    D.info['length'] = taper_length
    return D

def wide_line(length=5, WGwidth = 1, taper_width=0.5, taper_length = 3, layer = 0):
    """ Creates an taper"""
    D = Device('waveguide')
    D.add_polygon( [(0, WGwidth/2), (taper_length, taper_width/2), (length+taper_length, taper_width/2), (length+taper_length*2, WGwidth/2), (length+taper_length*2, -WGwidth/2), (length+taper_length, -taper_width/2), (taper_length, -taper_width/2), (0, -WGwidth/2)] )
    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [length+taper_length*2,0], width = WGwidth, orientation = 0)
    D.info['length'] = taper_length
    return D

def ring(radius , width, layer = 0):
    ring = Device('bend90 waveguide')
    E3 = pg.arc(radius = radius, width = width, theta = 360, start_angle = 0, angle_resolution = 1.5,layer=layer)
    ring.add_ref(E3)
    ring.add_port(name = 'wgport1', midpoint = [radius,0],  orientation = 180)
    ring.add_port(name = 'wgport2', midpoint = [0,radius],  orientation = 180)
    return ring
def cw_bend(radius = 1, width = 0.5, start_angle = 90,theta=90, layer = 0):
    cw = Device('cw_bend waveguide')
    E3 = pg.arc(radius = radius, width = width, theta = -theta, start_angle = start_angle, angle_resolution = 1,layer=layer)
    cw.add_ref(E3)
    cw.add_port(name = 'wgport1', midpoint = [radius*cos((start_angle)*pi/180),radius*sin((start_angle)*pi/180)], width = width,  orientation = start_angle+90)
    cw.add_port(name = 'wgport2', midpoint = [-radius*sin((start_angle-theta-90)*pi/180),radius*cos((start_angle-theta-90)*pi/180)], width = width,  orientation = start_angle-theta-90)
    cw.info['radius'] = radius
    cw.info['width'] = width
    return cw
def ccw_bend(radius = 1, width = 0.5, start_angle = 90,theta=90, layer = 0):    
    ccw = Device('ccw_bend waveguide')
    E3 = pg.arc(radius = radius, width = width, theta = theta, start_angle = start_angle, angle_resolution = 1,layer=layer)

    ccw.add_ref(E3)
    ccw.add_port(name = 'wgport1', midpoint = [radius*cos((start_angle)*pi/180),radius*sin((start_angle)*pi/180)], width = width,  orientation = start_angle-90)
    ccw.add_port(name = 'wgport2', midpoint = [radius*sin((start_angle+theta+90)*pi/180),-radius*cos((start_angle+theta+90)*pi/180)], width = width,  orientation = start_angle+theta+90)
    ccw.info['radius'] = radius
    ccw.info['width'] = width

    return ccw
def euler_bend(radius = 1, width = 0.5, theta = 90,p=0.2,use_eff=False, layer = 0,num_pts=720):
    P= Path()
    P.append( pp.euler(radius = radius, angle = theta,p=p,use_eff=use_eff,num_pts=720) )
    X = CrossSection().add(width = width, offset = 0, layer = 0, ports = ('wgport1','wgport2'))
    euler=P.extrude(X)
    euler.info['radius'] = radius
    euler.info['width'] = width
    euler.info['length'] = P.length()
    return euler

def s_bend(x=10,radius = 1, width = 0.5, turn = 2,p=0.2,use_eff=False, layer = 0,num_pts=720, bendtype="euler"):
    P= Path()
    P.append( pp.straight(length = x))

    if bendtype=="euler":
        for i in range(0,turn):
            P.append( pp.euler(radius = radius, angle = -180,p=p,use_eff=use_eff,num_pts=720) )
            P.append( pp.straight(length = x))
            P.append( pp.euler(radius = radius, angle = 180,p=p,use_eff=use_eff,num_pts=720) )
            P.append( pp.straight(length = x))
    else:
        for i in range(0,turn):
            P.append( pp.arc(radius = radius, angle = -180,num_pts=720) )
            P.append( pp.straight(length = x))
            P.append( pp.arc(radius = radius, angle = 180,num_pts=720) )
            P.append( pp.straight(length = x))
    X = CrossSection().add(width = width, offset = 0, layer = 0, ports = ('wgport1','wgport2'))
    s_bend=P.extrude(X)

    s_bend.info['radius'] = radius
    s_bend.info['length'] = P.length()
    return s_bend

def MMI11(length=100, width = 0.5,port_width=0.5, taper_length=10, taper_width=1.2):
    MMI11 = Device('MMI11')
    
    a= taper_length
    b= taper_length+length
    c= 2*taper_length+length
    #Left port
    MMI11.add_polygon( [(0, -port_width/2), (a, -taper_width/2), (a, taper_width/2), (0, port_width/2)] )
    #body
    MMI11.add_polygon( [(a, -width/2), (b, -width/2), (b, width/2), (a, width/2)] )
    #right port    
    MMI11.add_polygon( [(b, -taper_width/2), (c, -port_width/2), (c, port_width/2), (b, taper_width/2)] )
    
    MMI11.add_port(name = 'wgport1', midpoint = [0,0], width = port_width, orientation = 180)
    MMI11.add_port(name = 'wgport2', midpoint = [c,0], width = port_width, orientation = 0)
    return MMI11

def MMI22(length=100, width = 0.5,port_width=0.5, taper_length=10, taper_width=1.2,port_gap_ratio=0.5):
    MMI22 = Device('MMI22')
    
    
    a= taper_length
    b= taper_length+length
    c= 2*taper_length+length
    port_pos= width*port_gap_ratio/2
    #Left port
    MMI22.add_polygon( [(0, port_pos-port_width/2), (a, port_pos-taper_width/2), (a, port_pos+taper_width/2), (0, port_pos+port_width/2)] )
    MMI22.add_polygon( [(0, -port_pos-port_width/2), (a, -port_pos-taper_width/2), (a, -port_pos+taper_width/2), (0, -port_pos+port_width/2)] )
    #body
    MMI22.add_polygon( [(a, -width/2), (b, -width/2), (b, width/2), (a, width/2)] )
    #right port    
    MMI22.add_polygon( [(b, port_pos-taper_width/2), (c, port_pos-port_width/2), (c, port_pos+port_width/2), (b, port_pos+taper_width/2)] )
    MMI22.add_polygon( [(b, -port_pos-taper_width/2), (c, -port_pos-port_width/2), (c, -port_pos+port_width/2), (b, -port_pos+taper_width/2)] )
    
    MMI22.add_port(name = 'wgport1', midpoint = [0,port_pos], width = port_width, orientation = 180)
    MMI22.add_port(name = 'wgport2', midpoint = [0,-port_pos], width = port_width, orientation = 180)
    MMI22.add_port(name = 'wgport3', midpoint = [c,port_pos], width = port_width, orientation = 0)
    MMI22.add_port(name = 'wgport4', midpoint = [c,-port_pos], width = port_width, orientation = 0)
    return MMI22
def MMI12(length=100, width = 0.5,port_width=0.5, taper_length=2, taper_width=1.2,port_gap_ratio=0.5):
    MMI12 = Device('MMI12')
    
    a= taper_length
    b= taper_length+length
    c= 2*taper_length+length
    port_pos= width*port_gap_ratio/2
    #Left port
    MMI12.add_polygon( [(0, -port_width/2), (a, -taper_width/2), (a, taper_width/2), (0, port_width/2)] )
    #body
    MMI12.add_polygon( [(a, -width/2), (b, -width/2), (b, width/2), (a, width/2)] )
    #right port    
    MMI12.add_polygon( [(b, port_pos-taper_width/2), (c, port_pos-port_width/2), (c, port_pos+port_width/2), (b, port_pos+taper_width/2)] )
    MMI12.add_polygon( [(b, -port_pos-taper_width/2), (c, -port_pos-port_width/2), (c, -port_pos+port_width/2), (b, -port_pos+taper_width/2)] )
    
    MMI12.add_port(name = 'wgport1', midpoint = [0,0], width = port_width, orientation = 180)
    MMI12.add_port(name = 'wgport2', midpoint = [c,port_pos], width = port_width, orientation = 0)
    MMI12.add_port(name = 'wgport3', midpoint = [c,-port_pos], width = port_width, orientation = 0)
    return MMI12

def Beam_Splitter_DC(WGwidth = 1, coupling_length=100, coupling_gap = 0.5, port_gap12 = 301.1, port_gap34 = 301.1, buffer_length = 1000, a=0, b=0,ref_Length=1, layer = 0):
    D = Device('Beam spliter with DC')
    length=[0,0]
    for i in [1,-1]: # one for top one for bottom
        wg_L_port = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
        wg_L_port.y = i*port_gap12/2
        wg_L_port.xmin = 0

        wg_CL = D.add_ref(waveguide(length = coupling_length, width=WGwidth)) #coupling region 
        wg_CL.y= a + i*(coupling_gap+WGwidth)/2
        wg_CL.xmin = wg_L_port.xsize+buffer_length
  
        wg_R_port = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
        wg_R_port.y = b + i*port_gap34/2      
        wg_R_port.xmin = wg_L_port.xsize+buffer_length+wg_CL.xsize+buffer_length  
        D.add_ref( pr.route_basic(port1 = wg_CL.ports['wgport1'], port2 = wg_L_port.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
        er2=D.add_ref( pr.route_basic(port1 = wg_R_port.ports['wgport1'], port2 = wg_CL.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
        length[int(np.floor((1.5-i)/2))]=er2.info['length']
    length_differance = length[0]-length[1]
    D.add_port(name = 'wgport1', midpoint = [wg_L_port.xmin,port_gap12/2], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_L_port.xmin,-port_gap12/2], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport3', midpoint = [wg_R_port.xmax,b + port_gap34/2], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport4', midpoint = [wg_R_port.xmax,b - port_gap34/2], width = WGwidth, orientation =0)
    D.info['length_differance'] = length_differance
    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D  

def Beam_Splitter_MMI12(WGwidth = 1, taper_length=2, taper_width=1.2, MMI_coupling_length=100, MMI_coupling_width = 0.5,coupling_port_gap=0.5, port_gap34 = 301.1, buffer_length1 = 100, buffer_length = 1000, a=0, b=0,ref_Length=1, layer = 0):
    D = Device('Beam spliter with MMI')

    wg_L_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port1.y =0
    wg_L_port1.xmin = 0

    MMI = D.add_ref(MMI12(length=MMI_coupling_length, width = MMI_coupling_width ,port_width=WGwidth, taper_length=taper_length, taper_width=taper_width,port_gap_ratio=coupling_port_gap)) #coupling region 
    MMI.y= a
    MMI.xmin = wg_L_port1.xsize+buffer_length1

    wg_R_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
    wg_R_port1.y = b + port_gap34/2      
    wg_R_port1.xmin = wg_L_port1.xsize+buffer_length1+MMI.xsize+buffer_length 

    wg_R_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
    wg_R_port2.y = b - port_gap34/2      
    wg_R_port2.xmin = wg_L_port1.xsize+buffer_length1+MMI.xsize+buffer_length  


    D.add_ref( pr.route_basic(port1 = MMI.ports['wgport1'], port2 = wg_L_port1.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    R1_c=D.add_ref( pr.route_basic(port1 = wg_R_port1.ports['wgport1'], port2 = MMI.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    R2_c=D.add_ref( pr.route_basic(port1 = wg_R_port2.ports['wgport1'], port2 = MMI.ports['wgport3'], path_type = 'sine', width_type = 'sine'))
    

    
    length_differance = R1_c.info['length']-R2_c.info['length']

    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.xmin,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport3', midpoint = [wg_R_port1.xmax,b + port_gap34/2], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport4', midpoint = [wg_R_port2.xmax,b - port_gap34/2], width = WGwidth, orientation =0)
    D.info['length_differance'] = length_differance
    
    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D 

def Beam_Splitter_MMI22(WGwidth = 1, taper_length=10, taper_width=1.2, MMI_coupling_length=100, MMI_coupling_width = 0.5,coupling_port_gap=0.5, port_gap12 = 301.1, port_gap34 = 301.1, buffer_length = 1000, buffer_length2 = 1000, a=0, b=0,ref_Length=1, layer = 0):
    D = Device('Beam spliter with DC')

    wg_L_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port1.y = port_gap12/2
    wg_L_port1.xmin = 0
    wg_L_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port2.y = -port_gap12/2
    wg_L_port2.xmin = 0


    MMI = D.add_ref(MMI22(length=MMI_coupling_length, width = MMI_coupling_width, taper_length=taper_length,port_gap_ratio=coupling_port_gap, taper_width=taper_width,port_width=WGwidth)) #coupling region 
    MMI.y= a
    MMI.xmin = wg_L_port1.xmax+buffer_length
 
    wg_R_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
    wg_R_port1.y = b + port_gap34/2      
    wg_R_port1.xmin = wg_L_port1.xsize+buffer_length+MMI.xsize+buffer_length2  
    wg_R_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
    wg_R_port2.y = b - port_gap34/2      
    wg_R_port2.xmin = wg_L_port1.xsize+buffer_length+MMI.xsize+buffer_length2  


    D.add_ref( pr.route_basic(port1 = MMI.ports['wgport1'], port2 = wg_L_port1.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    D.add_ref( pr.route_basic(port1 = MMI.ports['wgport2'], port2 = wg_L_port2.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    er1=D.add_ref( pr.route_basic(port1 = wg_R_port1.ports['wgport1'], port2 = MMI.ports['wgport3'], path_type = 'sine', width_type = 'sine'))
    er2=D.add_ref( pr.route_basic(port1 = wg_R_port2.ports['wgport1'], port2 = MMI.ports['wgport4'], path_type = 'sine', width_type = 'sine'))

    length_differance = er1.info['length']-er2.info['length']
    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.xmin,wg_L_port1.y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_L_port2.xmin,wg_L_port2.y], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport3', midpoint = [wg_R_port1.xmax,wg_R_port1.y], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport4', midpoint = [wg_R_port2.xmax,wg_R_port2.y], width = WGwidth, orientation =0)
    D.info['length_differance'] = length_differance

    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D 

def Beam_Splitter_cMMI33(WGwidth = 1, taper_length=2, taper_width=1.2, MMI1_coupling_length=100, MMI1_coupling_width = 0.5,coupling_port_gap1=0.5, MMI2_coupling_length=100, MMI2_coupling_width = 0.5,coupling_port_gap2=0.5, port_gapL = 301.1, port_gapR = 301.1, buffer_center = 1000, buffer_length = 1000, ref_Length=1, layer = 0):
    D = Device('Beam spliter with cascade MMI')
# ports and MMI creation
    wg_L_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port1.y = port_gapL
    wg_L_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port2.y = 0
    wg_L_port3 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port3.y = -port_gapL

    MMI1 = D.add_ref(MMI22(length=MMI1_coupling_length, width = MMI1_coupling_width,port_width=WGwidth, taper_length=taper_length, taper_width=taper_width,port_gap_ratio=coupling_port_gap1))
    MMI1.y= -coupling_port_gap1*MMI1_coupling_width/2
    MMI1.x= wg_L_port1.xmax+buffer_length+MMI1_coupling_length/2
    MMI2 = D.add_ref(MMI22(length=MMI2_coupling_length, width = MMI2_coupling_width,port_width=WGwidth, taper_length=taper_length, taper_width=taper_width,port_gap_ratio=coupling_port_gap2))
    MMI2.y= coupling_port_gap2*MMI2_coupling_width/2
    MMI2.x= MMI1.xmax+buffer_center+MMI2_coupling_length/2

    wg_R_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_R_port1.y = port_gapR
    wg_R_port1.xmin = MMI2.xmax+buffer_length
    wg_R_port2 = D.add_ref(waveguide(length = ref_Length,width=WGwidth))  #left port
    wg_R_port2.y = 0
    wg_R_port2.xmin = MMI2.xmax+buffer_length
    wg_R_port3 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_R_port3.y = -port_gapR
    wg_R_port3.xmin = MMI2.xmax+buffer_length

# ports and MMI cennection   
    wg_L_port1_1 = D.add_ref(waveguide(length = MMI2.xmin-buffer_length+ref_Length, width=WGwidth))
    wg_L_port1_1.y = wg_L_port1.y
    wg_L_port1_1.xmin = wg_L_port1.xmax
    D.add_ref(pr.route_basic(port1 = wg_L_port2.ports['wgport2'], port2 = MMI1.ports['wgport1'], path_type = 'sine', width_type = 'sine')) 
    D.add_ref(pr.route_basic(port1 = wg_L_port3.ports['wgport2'], port2 = MMI1.ports['wgport2'], path_type = 'sine', width_type = 'sine')) 
    
    wg_R_port3_1 = D.add_ref(waveguide(length = MMI2.xmin-buffer_length+ref_Length, width=WGwidth))
    wg_R_port3_1.y = wg_R_port3.y
    wg_R_port3_1.xmin = MMI1.xmax+ buffer_length
    D.add_ref(pr.route_basic(port1 = wg_L_port1_1.ports['wgport2'], port2 = MMI2.ports['wgport1'], path_type = 'sine', width_type = 'sine')) 
    D.add_ref(pr.route_basic(port1 = MMI1.ports['wgport3'], port2 = MMI2.ports['wgport2'], path_type = 'sine', width_type = 'sine')) 
        
    D.add_ref(pr.route_basic(port1 = MMI1.ports['wgport4'], port2 = wg_R_port3_1.ports['wgport1'], path_type = 'sine', width_type = 'sine')) 
    D.add_ref(pr.route_basic(port1 = MMI2.ports['wgport3'], port2 = wg_R_port1.ports['wgport1'], path_type = 'sine', width_type = 'sine')) 
    D.add_ref(pr.route_basic(port1 = MMI2.ports['wgport4'], port2 = wg_R_port2.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
 
    
    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.xmin,wg_L_port1.y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_L_port2.xmin,wg_L_port2.y], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport3', midpoint = [wg_L_port3.xmin,wg_L_port3.y], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport4', midpoint = [wg_R_port1.xmax,wg_R_port1.y], width = WGwidth, orientation = 0)
    D.add_port(name = 'wgport5', midpoint = [wg_R_port2.xmax,wg_R_port2.y], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport6', midpoint = [wg_R_port3.xmax,wg_R_port3.y], width = WGwidth, orientation =0)

    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D 

def Beam_Splitter_mMMI33(WGwidth = 1,radius=200, taper_length=10, taper_width=1.2, MMI1_coupling_length=100, MMI1_coupling_width = 0.5,coupling_port_gap1=0.5, MMI2_coupling_length=100, MMI2_coupling_width = 0.5,coupling_port_gap2=0.5, port_gapL = 301.1, port_gapR = 301.1, buffer_center = 1000, buffer_length = 1000, ref_Length=1, layer = 0):
    D = Device('Beam spliter with cascade MMI')
# ports and MMI creation
    wg_L_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port1.y = port_gapL/2
    wg_L_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port2.y = -port_gapL/2
    wbend=euler_bend(radius=radius,width=WGwidth,theta=-180).xsize
    MMI1 = D.add_ref(MMI22(length=MMI1_coupling_length, width = MMI1_coupling_width,port_width=WGwidth, taper_length=taper_length, taper_width=taper_width,port_gap_ratio=coupling_port_gap1))
    MMI1.y= 0
    MMI1.x= wg_L_port1.xmax+buffer_length+MMI1_coupling_length/2
    MMI2 = D.add_ref(MMI22(length=MMI2_coupling_length, width = MMI2_coupling_width,port_width=WGwidth, taper_length=taper_length, taper_width=taper_width,port_gap_ratio=coupling_port_gap2))
    MMI2.y= 0
    MMI2.xmin= MMI1.xmax+buffer_center+wbend*2
    bend0=D.add_ref(euler_bend(radius=radius,width=WGwidth,theta=180))
    bend1=D.add_ref(euler_bend(radius=radius,width=WGwidth,theta=-180))
    bend2=D.add_ref(euler_bend(radius=radius,width=WGwidth,theta=180))
    bend3=D.add_ref(euler_bend(radius=radius,width=WGwidth,theta=180))
    

# ports and MMI cennection   
    D.add_ref(pr.route_basic(port1 = wg_L_port1.ports['wgport2'], port2 = MMI1.ports['wgport1'], path_type = 'sine', width_type = 'sine')) 
    D.add_ref(pr.route_basic(port1 = wg_L_port2.ports['wgport2'], port2 = MMI1.ports['wgport2'], path_type = 'sine', width_type = 'sine')) 
    bend0.connect(port='wgport1',destination=MMI2.ports['wgport3']) 
    bend1.connect(port='wgport1',destination=MMI1.ports['wgport4']) 
    bend2.connect(port='wgport1',destination=bend1.ports['wgport2'])
    bend3.connect(port='wgport1',destination=MMI2.ports['wgport2'])
 
    wg_R_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #Right port
    wg_R_port1.connect(port='wgport2',destination=bend0.ports['wgport2'])
    wg_R_port2 = D.add_ref(waveguide(length = ref_Length,width=WGwidth))  #Right port
    wg_R_port2.y = MMI2.ports['wgport4'].y
    wg_R_port2.xmin = MMI2.xmax+buffer_length
    wg_R_port3 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #Right port
    wg_R_port3.connect(port='wgport1',destination=bend3.ports['wgport2'])
    wg_R_port4 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #Right port
    wg_R_port4.connect(port='wgport1',destination=bend2.ports['wgport2'])

    D.add_ref(pr.route_basic(port1 = MMI1.ports['wgport3'], port2 = MMI2.ports['wgport1'], path_type = 'sine', width_type = 'sine')) 
 #    D.add_ref(pr.route_basic(port1 = bend0.ports['wgport2'], port2 = wg_R_port1.ports['wgport2'], path_type = 'sine', width_type = 'sine')) 
    D.add_ref(pr.route_basic(port1 = MMI2.ports['wgport4'], port2 = wg_R_port2.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
 #   D.add_ref(pr.route_basic(port1 = bend3.ports['wgport2'], port2 = wg_R_port3.ports['wgport1'], path_type = 'sine', width_type = 'sine')) 
 #   D.add_ref(pr.route_basic(port1 = bend2.ports['wgport2'], port2 = wg_R_port4.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
 
    
    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.xmin,wg_L_port1.y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_L_port2.xmin,wg_L_port2.y], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport3', midpoint = [wg_R_port1.xmin,wg_R_port1.y], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport4', midpoint = [wg_R_port2.xmax,wg_R_port2.y], width = WGwidth, orientation = 0)
    D.add_port(name = 'wgport5', midpoint = [wg_R_port3.xmax,wg_R_port3.y], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport6', midpoint = [wg_R_port4.xmax,wg_R_port4.y], width = WGwidth, orientation =0)

    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D 

def sin_taper_grat(period=0.5, WGwidth = 1, sin_width=0.5, teeth_num = 3, sin_resolution = 20,start_angle=90, layer = 0):
    """ Creates an Sin grating taper"""

    sin_width1 = sin_width
    t = np.linspace(0, period*teeth_num, np.ceil(teeth_num * sin_resolution+1))
    
    sin_width = sin_width1/period/teeth_num * t
    upper_points_x = (t).tolist()
    upper_points_y = (sin_width*cos(2*np.pi/period*t+start_angle*np.pi/180)+WGwidth/2).tolist()
    lower_points_x = (t).tolist()
    lower_points_y = (sin_width*cos(2*np.pi/period*t+np.pi+start_angle*np.pi/180)-WGwidth/2).tolist()
    xpts = upper_points_x + lower_points_x[::-1]
    ypts = upper_points_y + lower_points_y[::-1]

    D = Device('sin_grat')
    D.add_polygon(points = (xpts,ypts), layer = layer)    
    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [period*teeth_num,0], width = WGwidth, orientation = 0)
    D.info['length'] = period*teeth_num
    return D
def sin_grat(period=0.5, WGwidth = 1, sin_width=0.5, teeth_num = 3, sin_resolution = 20,start_angle=90, layer = 0):
    """ Creates an Sin grating """
   
    t = np.linspace(0, period*teeth_num, np.ceil(teeth_num * sin_resolution+1))
    upper_points_x = (t).tolist()
    upper_points_y = (sin_width*cos(2*np.pi/period*t+start_angle*np.pi/180)+WGwidth/2).tolist()
    lower_points_x = (t).tolist()
    lower_points_y = (sin_width*cos(2*np.pi/period*t+np.pi+start_angle*np.pi/180)-WGwidth/2).tolist()
    xpts = upper_points_x + lower_points_x[::-1]
    ypts = upper_points_y + lower_points_y[::-1]

    D = Device('sin_grat')
    D.add_polygon(points = (xpts,ypts), layer = layer)
    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [period*teeth_num,0], width = WGwidth, orientation = 0)
    D.info['length'] = period*teeth_num
    return D
def rect_grat(period=0.5, fill_factor = 0.5, WGwidth = 1, grating_width=0.5, teeth_num = 1, start_width=1, layer = 0):
    """ Creates an Sin grating """
    wide_width = WGwidth+grating_width
    narrow_width = WGwidth-grating_width
    D = Device('sin_grat')
    for i in range(1,teeth_num+1):
    
        D.add_polygon([((i-1)*period, wide_width/2), ((i-1)*period+period*fill_factor, wide_width/2), ((i-1)*period+period*fill_factor, -wide_width/2), ((i-1)*period, -wide_width/2)])
        if WGwidth > grating_width:
            D.add_polygon([((i)*period, narrow_width/2), ((i-1)*period+period*fill_factor, narrow_width/2), ((i-1)*period+period*fill_factor, -narrow_width/2), ((i)*period, -narrow_width/2)])

    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [period*teeth_num,0], width = WGwidth, orientation = 0)
    D.info['length'] = period*teeth_num
    return D
def Delay_Line(WGwidth = 1, turn_num=4, r_bending= 1000, a=50, b=50, wt_gap=50, layer = 0):
    
    DL=0
    wt_gap=WGwidth+wt_gap
    #minimum wt_b  ->> r_bending*4 +wt_gap*18
    wt_a = r_bending*4 +wt_gap*2*(turn_num-1)+a
    #minimum wt_b  ->> r_bending*2 +wt_gap*20
    wt_b = r_bending*2 +wt_gap*2*turn_num+b

    D1 = Device() # device reset 
    #----------------clockwise waveguide
    
    wg = D1.add_ref(waveguide(length=r_bending, width = WGwidth))
    D1.add_port(port = wg.ports['wgport2'], name = 1)
    DL=DL+wg.xsize

    num1=0
    for num1 in range(1,turn_num+1):
    
        wg = D1.add_ref(waveguide(length=wt_a-2*r_bending-(2*num1-3)*wt_gap, width = WGwidth))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D1.ports[4*num1-3])
        D1.add_port(port = wg.ports['wgport2'], name = 4*num1-2)
        
        bwg = D1.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D1.ports[4*num1-2])
        D1.add_port(port = bwg.ports['wgport1'], name = 4*num1-1)
    
        wg = D1.add_ref(waveguide(length=wt_b-2*r_bending-(2*num1-1)*wt_gap, width = WGwidth))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D1.ports[4*num1-1])
        D1.add_port(port = wg.ports['wgport2'], name = 4*num1)
        
        bwg = D1.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D1.ports[4*num1])
        D1.add_port(port = bwg.ports['wgport1'], name = 4*num1+1)
    
       
    
     #----------------center conversion waveguide
    num1=num1+1    
    wg = D1.add_ref(waveguide(length=wt_a/2-2*r_bending-(num1-2)*wt_gap, width = WGwidth))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D1.ports[4*num1-3])
    D1.add_port(port = wg.ports['wgport2'], name = 4*num1-2)
        
    bwg = D1.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+r_bending*pi/2
    bwg.connect(port = 'wgport2', destination = D1.ports[4*num1-2])
    D1.add_port(port = bwg.ports['wgport1'], name = 4*num1-1)
    
    wg = D1.add_ref(waveguide(length=wt_b-2*r_bending-(2*num1-2)*wt_gap, width = WGwidth))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D1.ports[4*num1-1])
    D1.add_port(port = wg.ports['wgport2'], name = 4*num1)
        
    bwg = D1.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+r_bending*pi/2
    bwg.connect(port = 'wgport1', destination = D1.ports[4*num1])
    D1.add_port(port = bwg.ports['wgport2'], name = 4*num1+1)    
    
    wg = D1.add_ref(waveguide(length=wt_a/2-2*r_bending-(num1-2)*wt_gap, width = WGwidth))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D1.ports[4*num1+1])
    D1.add_port(port = wg.ports['wgport2'], name = 4*num1+2)
    
    #----------------counter clockwise waveguide
    D2 = Device()
    bwg = D2.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+r_bending*pi/2
    D2.add_port(port = bwg.ports['wgport1'], name = 1)
    D2.add_port(port = bwg.ports['wgport2'], name = 0)
    num2=0
    for num2 in range(1,turn_num+1):
    
        wg = D2.add_ref(waveguide(length=wt_a-2*r_bending-(2*num2-3)*wt_gap, width = WGwidth))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D2.ports[4*num2-3])
        D2.add_port(port = wg.ports['wgport2'], name = 4*num2-2)
        
        bwg = D2.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D2.ports[4*num2-2])
        D2.add_port(port = bwg.ports['wgport1'], name = 4*num2-1)
    
        wg = D2.add_ref(waveguide(length=wt_b-2*r_bending-(2*num2-1)*wt_gap, width = WGwidth))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D2.ports[4*num2-1])
        D2.add_port(port = wg.ports['wgport2'], name = 4*num2)
        
        bwg = D2.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D2.ports[4*num2])
        D2.add_port(port = bwg.ports['wgport1'], name = 4*num2+1)
   
    wg = D2.add_ref(waveguide(length=wt_b-2*r_bending, width = WGwidth))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D2.ports[0])
    D2.add_port(port = wg.ports['wgport2'], name = 4*num2+2)
    
    bwg = D2.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+r_bending*pi/2
    bwg.connect(port = 'wgport2', destination = D2.ports[4*num2+2])
    D2.add_port(port = bwg.ports['wgport1'], name = 4*num2+3)
   

    
    half=D1.add_ref(D2)
    half.connect(port = 4*num2+1, destination = D1.ports[4*num1+2])
    D=Device()
    D.add_ref(D1)  
    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [D.xsize,0], width = WGwidth, orientation = 0)
    D.info['Delay line length'] = DL/10000
    return D
def Modified_Delay_Line(WGwidth = 1, taper_width=3, taper_length = 3, turn_num=4, r_bending= 1000, a=50, b=50, wt_gap=50, layer = 0):
    
    DL=0
    wt_gap=WGwidth+wt_gap
    #minimum wt_b  ->> r_bending*4 +wt_gap*18
    wt_a = r_bending*4 +wt_gap*2*(turn_num-1)+2*taper_length+a
    #minimum wt_b  ->> r_bending*2 +wt_gap*20
    wt_b = r_bending*2 +wt_gap*2*turn_num+b

    D1 = Device() # device reset 
    #----------------clockwise waveguide
    
    wg = D1.add_ref(waveguide(length=r_bending, width = WGwidth))
    D1.add_port(port = wg.ports['wgport2'], name = 1)
    DL=DL+wg.xsize

    num1=0
    for num1 in range(1,turn_num+1):
        wg = D1.add_ref(wide_line(length=wt_a-2*r_bending-(2*num1-3)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D1.ports[4*num1-3])
        D1.add_port(port = wg.ports['wgport2'], name = 4*num1-2)
        
        bwg = D1.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D1.ports[4*num1-2])
        D1.add_port(port = bwg.ports['wgport1'], name = 4*num1-1)
    
        wg = D1.add_ref(wide_line(length=wt_b-2*r_bending-(2*num1-1)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D1.ports[4*num1-1])
        D1.add_port(port = wg.ports['wgport2'], name = 4*num1)
        
        bwg = D1.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D1.ports[4*num1])
        D1.add_port(port = bwg.ports['wgport1'], name = 4*num1+1)
    
       
    
     #----------------center conversion waveguide
    num1=num1+1    
    wg = D1.add_ref(wide_line(length=wt_a/2-2*r_bending-(num1-2)*wt_gap-taper_length, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D1.ports[4*num1-3])
    D1.add_port(port = wg.ports['wgport2'], name = 4*num1-2)
        
    bwg = D1.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+r_bending*pi/2
    bwg.connect(port = 'wgport2', destination = D1.ports[4*num1-2])
    D1.add_port(port = bwg.ports['wgport1'], name = 4*num1-1)
    
    wg = D1.add_ref(wide_line(length=wt_b-2*r_bending-(2*num1-2)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D1.ports[4*num1-1])
    D1.add_port(port = wg.ports['wgport2'], name = 4*num1)
        
    bwg = D1.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+r_bending*pi/2
    bwg.connect(port = 'wgport1', destination = D1.ports[4*num1])
    D1.add_port(port = bwg.ports['wgport2'], name = 4*num1+1)    
    
    wg = D1.add_ref(wide_line(length=wt_a/2-2*r_bending-(num1-2)*wt_gap-taper_length, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D1.ports[4*num1+1])
    D1.add_port(port = wg.ports['wgport2'], name = 4*num1+2)
    
    #----------------counter clockwise waveguide
    D2 = Device()
    bwg = D2.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+r_bending*pi/2
    D2.add_port(port = bwg.ports['wgport1'], name = 1)
    D2.add_port(port = bwg.ports['wgport2'], name = 0)
    num2=0
    for num2 in range(1,turn_num+1):
    
        wg = D2.add_ref(wide_line(length=wt_a-2*r_bending-(2*num2-3)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D2.ports[4*num2-3])
        D2.add_port(port = wg.ports['wgport2'], name = 4*num2-2)
        
        bwg = D2.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D2.ports[4*num2-2])
        D2.add_port(port = bwg.ports['wgport1'], name = 4*num2-1)
    
        wg = D2.add_ref(wide_line(length=wt_b-2*r_bending-(2*num2-1)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D2.ports[4*num2-1])
        D2.add_port(port = wg.ports['wgport2'], name = 4*num2)
        
        bwg = D2.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D2.ports[4*num2])
        D2.add_port(port = bwg.ports['wgport1'], name = 4*num2+1)
   
    wg = D2.add_ref(wide_line(length=wt_b-2*r_bending, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D2.ports[0])
    D2.add_port(port = wg.ports['wgport2'], name = 4*num2+2)
    
    bwg = D2.add_ref(ccw_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+r_bending*pi/2
    bwg.connect(port = 'wgport2', destination = D2.ports[4*num2+2])
    D2.add_port(port = bwg.ports['wgport1'], name = 4*num2+3)
   

    
    half=D1.add_ref(D2)
    half.connect(port = 4*num2+1, destination = D1.ports[4*num1+2])
 
    D=Device()
    D.add_ref(D1)  
    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [D.xsize,0], width = WGwidth, orientation = 0)
    D.info['Delay line length'] = DL/10000
    return D    

def Modified_euler_Delay_Line(WGwidth = 1, taper_width=3, taper_length = 3, turn_num=4, r_bending= 1000, a=50, b=50, wt_gap=50, layer = 0):
    
    DL=0
    wt_gap=WGwidth+wt_gap
    euler_r_bending=euler_bend(radius = r_bending, width = WGwidth,theta=90).xsize
    #minimum wt_b  ->> r_bending*4 +wt_gap*18
    wt_a = euler_r_bending*4 +wt_gap*2*(turn_num-1)+2*taper_length+a
    #minimum wt_b  ->> r_bending*2 +wt_gap*20
    wt_b = euler_r_bending*2 +wt_gap*2*turn_num+b

    D1 = Device() # device reset 
    #----------------clockwise waveguide
    
    wg = D1.add_ref(waveguide(length=euler_r_bending, width = WGwidth))
    D1.add_port(port = wg.ports['wgport2'], name = 1)
    DL=DL+wg.xsize

    num1=0
    for num1 in range(1,turn_num+1):
        wg = D1.add_ref(wide_line(length=wt_a-2*euler_r_bending-(2*num1-3)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D1.ports[4*num1-3])
        D1.add_port(port = wg.ports['wgport2'], name = 4*num1-2)
        
        bwg = D1.add_ref(euler_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+euler_r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D1.ports[4*num1-2])
        D1.add_port(port = bwg.ports['wgport1'], name = 4*num1-1)
    
        wg = D1.add_ref(wide_line(length=wt_b-2*euler_r_bending-(2*num1-1)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D1.ports[4*num1-1])
        D1.add_port(port = wg.ports['wgport2'], name = 4*num1)
        
        bwg = D1.add_ref(euler_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+euler_r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D1.ports[4*num1])
        D1.add_port(port = bwg.ports['wgport1'], name = 4*num1+1)
    
       
    
     #----------------center conversion waveguide
    num1=num1+1    
    wg = D1.add_ref(wide_line(length=wt_a/2-2*euler_r_bending-(num1-2)*wt_gap-taper_length, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D1.ports[4*num1-3])
    D1.add_port(port = wg.ports['wgport2'], name = 4*num1-2)
        
    bwg = D1.add_ref(euler_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+euler_r_bending*pi/2
    bwg.connect(port = 'wgport2', destination = D1.ports[4*num1-2])
    D1.add_port(port = bwg.ports['wgport1'], name = 4*num1-1)
    
    wg = D1.add_ref(wide_line(length=wt_b-2*euler_r_bending-(2*num1-2)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D1.ports[4*num1-1])
    D1.add_port(port = wg.ports['wgport2'], name = 4*num1)
        
    bwg = D1.add_ref(euler_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+euler_r_bending*pi/2
    bwg.connect(port = 'wgport1', destination = D1.ports[4*num1])
    D1.add_port(port = bwg.ports['wgport2'], name = 4*num1+1)    
    
    wg = D1.add_ref(wide_line(length=wt_a/2-2*euler_r_bending-(num1-2)*wt_gap-taper_length, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D1.ports[4*num1+1])
    D1.add_port(port = wg.ports['wgport2'], name = 4*num1+2)
    
    #----------------counter clockwise waveguide
    D2 = Device()
    bwg = D2.add_ref(euler_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+euler_r_bending*pi/2
    D2.add_port(port = bwg.ports['wgport1'], name = 1)
    D2.add_port(port = bwg.ports['wgport2'], name = 0)
    num2=0
    for num2 in range(1,turn_num+1):
    
        wg = D2.add_ref(wide_line(length=wt_a-2*euler_r_bending-(2*num2-3)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D2.ports[4*num2-3])
        D2.add_port(port = wg.ports['wgport2'], name = 4*num2-2)
        
        bwg = D2.add_ref(euler_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+euler_r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D2.ports[4*num2-2])
        D2.add_port(port = bwg.ports['wgport1'], name = 4*num2-1)
    
        wg = D2.add_ref(wide_line(length=wt_b-2*euler_r_bending-(2*num2-1)*wt_gap, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
        DL=DL+wg.xsize
        wg.connect(port = 'wgport1', destination = D2.ports[4*num2-1])
        D2.add_port(port = wg.ports['wgport2'], name = 4*num2)
        
        bwg = D2.add_ref(euler_bend(radius = r_bending, width = WGwidth,theta=90))
        DL=DL+euler_r_bending*pi/2
        bwg.connect(port = 'wgport2', destination = D2.ports[4*num2])
        D2.add_port(port = bwg.ports['wgport1'], name = 4*num2+1)
   
    wg = D2.add_ref(wide_line(length=wt_b-2*euler_r_bending, WGwidth = WGwidth, taper_width=taper_width, taper_length = taper_length))
    DL=DL+wg.xsize
    wg.connect(port = 'wgport1', destination = D2.ports[0])
    D2.add_port(port = wg.ports['wgport2'], name = 4*num2+2)
    
    bwg = D2.add_ref(euler_bend(radius = r_bending, width = WGwidth,theta=90))
    DL=DL+euler_r_bending*pi/2
    bwg.connect(port = 'wgport2', destination = D2.ports[4*num2+2])
    D2.add_port(port = bwg.ports['wgport1'], name = 4*num2+3)
   

    
    half=D1.add_ref(D2)
    half.connect(port = 4*num2+1, destination = D1.ports[4*num1+2])
 
    D=Device()
    D.add_ref(D1)  
    D.add_port(name = 'wgport1', midpoint = [0,0], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [D.xsize,0], width = WGwidth, orientation = 0)
    D.info['Delay line length'] = DL/10000
    return D   

def rotate_BS_MMI22(WGwidth = 1, taper_length=10, taper_width=1.2, R_angle=30,R_bend=300, MMI_coupling_length=100, MMI_coupling_width = 0.5,coupling_port_gap=0.5, port_gap12 = 301.1, port_gap34 = 301.1, buffer_length = 1000, a=0, b=0,ref_Length=1, layer = 0):
    D = Device('Beam spliter with DC')

    wg_L_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port1.y = port_gap12/2
    wg_L_port1.xmin = 0
    wg_L_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port2.y = -port_gap12/2
    wg_L_port2.xmin = 0

    R1 = D.add_ref(cw_bend(radius =R_bend+MMI_coupling_width*coupling_port_gap, width = WGwidth,theta=R_angle))
    R1.xmin = wg_L_port1.xmax+buffer_length
    R1.ymax= a+MMI_coupling_width*coupling_port_gap/2+WGwidth/2

    R2 = D.add_ref(cw_bend(radius = R_bend, width = WGwidth,theta=R_angle))
    R2.xmin = wg_L_port1.xmax+buffer_length
    R2.ymax= a-MMI_coupling_width*coupling_port_gap/2+WGwidth/2

    MMI = D.add_ref(MMI22(length=MMI_coupling_length, width = MMI_coupling_width,port_gap_ratio=coupling_port_gap, taper_length=taper_length, taper_width=taper_width, port_width=WGwidth)) #coupling region 
    MMI.y= a
    MMI.xmin = wg_L_port1.xmax+buffer_length

    R3 = D.add_ref(cw_bend(radius =R_bend+MMI_coupling_width*coupling_port_gap, width = WGwidth,theta=R_angle))
    R3.xmin = wg_L_port1.xmax+buffer_length
    R3.ymax= a+MMI_coupling_width*coupling_port_gap/2+WGwidth/2

    R4 = D.add_ref(cw_bend(radius = R_bend, width = WGwidth,theta=R_angle))
    R4.xmin = wg_L_port1.xmax+buffer_length
    R4.ymax= a-MMI_coupling_width*coupling_port_gap/2+WGwidth/2

    MMI.connect(port='wgport1', destination=R1.ports['wgport2'])
    R3.connect(port='wgport2',destination=MMI.ports['wgport4'])
    R4.connect(port='wgport2',destination=MMI.ports['wgport3'])
 
    
    wg_R_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
    wg_R_port1.y = b + port_gap34/2 - 2*R_bend*(1-cos(R_angle*pi/180))-sin(R_angle*pi/180)*(MMI_coupling_length+taper_length*2)
    wg_R_port1.xmin = R3.xmax+buffer_length  
    wg_R_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
    wg_R_port2.y = b - port_gap34/2 - 2*R_bend*(1-cos(R_angle*pi/180))-sin(R_angle*pi/180)*(MMI_coupling_length+taper_length*2)      
    wg_R_port2.xmin = R3.xmax+buffer_length

    D.add_ref( pr.route_basic(port1 = R1.ports['wgport1'], port2 = wg_L_port1.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    D.add_ref( pr.route_basic(port1 = R2.ports['wgport1'], port2 = wg_L_port2.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    D.add_ref( pr.route_basic(port1 = wg_R_port1.ports['wgport1'], port2 = R4.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    D.add_ref( pr.route_basic(port1 = wg_R_port2.ports['wgport1'], port2 = R3.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    
    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.xmin,wg_L_port1.y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_L_port2.xmin,wg_L_port2.y], width = WGwidth, orientation =180)    
    D.add_port(name = 'wgport3', midpoint = [wg_R_port1.xmax,wg_R_port1.y], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport4', midpoint = [wg_R_port2.xmax,wg_R_port2.y], width = WGwidth, orientation =0)
    return D 
def rotate_BS_MMI22_diagonal(WGwidth = 1, taper_length=10, taper_width=1.2, R_angle=30,R_bend=300, MMI_coupling_length=100, MMI_coupling_width = 0.5,coupling_port_gap=0.5, port_gap12 = 301.1, port_gap34 = 301.1, buffer_length = 1000, a=0, b=0,ref_Length=1, layer = 0):
    D = Device('Beam spliter with DC')

    wg_L_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port1.y = port_gap12/2
    wg_L_port1.xmin = 0
    wg_L_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port2.y = -port_gap12/2
    wg_L_port2.xmin = 0

    R1 = D.add_ref(cw_bend(radius =R_bend+MMI_coupling_width*coupling_port_gap, width = WGwidth,theta=R_angle))
    R1.xmin = wg_L_port1.xmax+buffer_length
    R1.ymax= a+MMI_coupling_width*coupling_port_gap/2+WGwidth/2

    R2 = D.add_ref(cw_bend(radius = R_bend, width = WGwidth,theta=R_angle))
    R2.xmin = wg_L_port1.xmax+buffer_length
    R2.ymax= a-MMI_coupling_width*coupling_port_gap/2+WGwidth/2
  

    MMI = D.add_ref(MMI22(length=MMI_coupling_length, width = MMI_coupling_width, taper_length=taper_length,port_gap_ratio=coupling_port_gap, taper_width=taper_width, port_width=WGwidth)) #coupling region 
    MMI.y= a
    MMI.xmin = wg_L_port1.xmax+buffer_length

    R3 = D.add_ref(cw_bend(radius =R_bend+MMI_coupling_width*coupling_port_gap, width = WGwidth,theta=abs(45-R_angle)))
    R3.xmin = wg_L_port1.xmax+buffer_length
    R3.ymax= a+MMI_coupling_width*coupling_port_gap/2+WGwidth/2

    R4 = D.add_ref(cw_bend(radius = R_bend, width = WGwidth,theta=abs(45-R_angle)))
    R4.xmin = wg_L_port1.xmax+buffer_length
    R4.ymax= a-MMI_coupling_width*coupling_port_gap/2+WGwidth/2

    MMI.connect(port='wgport1', destination=R1.ports['wgport2'])
    if R_angle<=45:
        R3.connect(port='wgport1',destination=MMI.ports['wgport3'])
        R4.connect(port='wgport1',destination=MMI.ports['wgport4'])
        D.add_port(name = 'wgport3', midpoint = [R3.ports['wgport2'].x,R3.ports['wgport2'].y], width = WGwidth, orientation =-45)
        D.add_port(name = 'wgport4', midpoint = [R4.ports['wgport2'].x,R4.ports['wgport2'].y], width = WGwidth, orientation =-45)

    else:
        R3.connect(port='wgport2',destination=MMI.ports['wgport4'])
        R4.connect(port='wgport2',destination=MMI.ports['wgport3'])
        D.add_port(name = 'wgport4', midpoint = [R3.ports['wgport1'].x,R3.ports['wgport1'].y], width = WGwidth, orientation =-45)
        D.add_port(name = 'wgport3', midpoint = [R4.ports['wgport1'].x,R4.ports['wgport1'].y], width = WGwidth, orientation =-45)

    

    D.add_ref( pr.route_basic(port1 = R1.ports['wgport1'], port2 = wg_L_port1.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    D.add_ref( pr.route_basic(port1 = R2.ports['wgport1'], port2 = wg_L_port2.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
  
    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.xmin,wg_L_port1.y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_L_port2.xmin,wg_L_port2.y], width = WGwidth, orientation =180)    
    return D 
def rotate_PBS_MMI22(WGwidth = 1, taper_length=10, taper_width=1,Lx =0,dLx = 46.08441, Ly =0, dLy =45.4379, R_angle=30,R_bend=300, MMI_coupling_length=100, MMI_coupling_width = 0.5,coupling_port_gap=0.5, port_gap12 = 301.1, port_gap34 = 301.1, buffer_length = 1000, a=0, b=0,ref_Length=1, layer = 0):
    D=Device('2by2 PBS_MZI_with MMI')
   
    L_MMI=D.add_ref(rotate_BS_MMI22_diagonal(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width,R_angle=R_angle,R_bend=R_bend, MMI_coupling_length=MMI_coupling_length, MMI_coupling_width = MMI_coupling_width, port_gap12 = port_gap12, port_gap34 = port_gap34, buffer_length = buffer_length, a=a, b=b))
    
    
    UR1=D.add_ref(ccw_bend(radius = R_bend, width = WGwidth,theta=135))
    UR1.connect(port='wgport1',destination=L_MMI.ports['wgport3'])
    US1=D.add_ref(waveguide(length=Ly,width=WGwidth))
    US1.connect(port='wgport1',destination=UR1.ports['wgport2'])
    UR2=D.add_ref(cw_bend(radius = R_bend, width = WGwidth,theta=180))
    UR2.connect(port='wgport1',destination=US1.ports['wgport2'])
    US2=D.add_ref(waveguide(length=Ly,width=WGwidth))
    US2.connect(port='wgport1',destination=UR2.ports['wgport2'])
    UR3=D.add_ref(ccw_bend(radius = R_bend, width = WGwidth,theta=90))
    UR3.connect(port='wgport1',destination=US2.ports['wgport2'])
    US3=D.add_ref(waveguide(length=Lx+dLx,width=WGwidth))
    US3.connect(port='wgport1',destination=UR3.ports['wgport2'])
    UR4=D.add_ref(cw_bend(radius = R_bend, width = WGwidth,theta=180))
    UR4.connect(port='wgport1',destination=US3.ports['wgport2'])
    US4=D.add_ref(waveguide(length=Lx+dLx,width=WGwidth))
    US4.connect(port='wgport1',destination=UR4.ports['wgport2'])
    UR5=D.add_ref(ccw_bend(radius = R_bend, width = WGwidth,theta=135))
    UR5.connect(port='wgport1',destination=US4.ports['wgport2'])
    
    LR1=D.add_ref(cw_bend(radius = R_bend, width = WGwidth,theta=135))
    LR1.connect(port='wgport1',destination=L_MMI.ports['wgport4'])
    LS1=D.add_ref(waveguide(length=Lx,width=WGwidth))
    LS1.connect(port='wgport1',destination=LR1.ports['wgport2'])
    LR2=D.add_ref(ccw_bend(radius = R_bend, width = WGwidth,theta=180))
    LR2.connect(port='wgport1',destination=LS1.ports['wgport2'])
    LS2=D.add_ref(waveguide(length=Lx,width=WGwidth))
    LS2.connect(port='wgport1',destination=LR2.ports['wgport2'])
    LR3=D.add_ref(cw_bend(radius = R_bend, width = WGwidth,theta=90))
    LR3.connect(port='wgport1',destination=LS2.ports['wgport2'])
    LS3=D.add_ref(waveguide(length=Ly+dLy,width=WGwidth))
    LS3.connect(port='wgport1',destination=LR3.ports['wgport2'])
    LR4=D.add_ref(ccw_bend(radius = R_bend, width = WGwidth,theta=180))
    LR4.connect(port='wgport1',destination=LS3.ports['wgport2'])
    LS4=D.add_ref(waveguide(length=Ly+dLy,width=WGwidth))
    LS4.connect(port='wgport1',destination=LR4.ports['wgport2'])
    LR5=D.add_ref(cw_bend(radius = R_bend, width = WGwidth,theta=135))
    LR5.connect(port='wgport1',destination=LS4.ports['wgport2'])
    
    R_MMI=D.add_ref(rotate_BS_MMI22_diagonal(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width,R_angle=R_angle,R_bend=R_bend, MMI_coupling_length=MMI_coupling_length, MMI_coupling_width = MMI_coupling_width, port_gap12 = port_gap12, port_gap34 = port_gap34, buffer_length = buffer_length, a=a, b=b))
    R_MMI.connect(port = 'wgport4', destination = UR5.ports['wgport2'])
    D.add_port(name = 'wgport1', midpoint = L_MMI.ports['wgport1'].midpoint, width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = L_MMI.ports['wgport2'].midpoint, width = WGwidth, orientation =180)    
    D.add_port(name = 'wgport3', midpoint = R_MMI.ports['wgport2'].midpoint, width = WGwidth, orientation =0)
    D.add_port(name = 'wgport4', midpoint = R_MMI.ports['wgport1'].midpoint, width = WGwidth, orientation =0)
     
    return D

def Electrode_xline(WG, gap=5, width=5,layer=2):
    D = Device('electrode')
    U_e=D.add_ref(waveguide(width=width,length=WG.xsize,layer=layer))
    L_e=D.add_ref(waveguide(width=width,length=WG.xsize,layer=layer))
    U_e.x=WG.x
    U_e.ymin=WG.ymax+gap-WG.ysize/2
    L_e.x=WG.x
    L_e.ymax=WG.ymin-gap+WG.ysize/2
   
    return D 

def Electrode_cwring(ring, start_angle = 30,theta=60, gap=5, width=5,layer=2):
    D = Device('electrode')
    U_e=D.add_ref(cw_bend(radius = ring.info['radius']+gap+width/2+ring.info['width']/2, width =width, start_angle =start_angle,theta=theta, layer = layer))
    L_e=D.add_ref(cw_bend(radius = ring.info['radius']-gap-width/2-ring.info['width']/2, width =width, start_angle =start_angle,theta=theta, layer = layer))
    U_e.origin=ring.origin
    L_e.origin=ring.origin

    return D 

def MZI_1by1(WGwidth = 1, common_length=200,taper_length=2, taper_width=1.2, MMI_coupling_length=100, MMI_coupling_width = 0.5,coupling_port_gap=0.5, port_gap34 = 301.1,buffer_length1 = 100,buffer_length = 1000, a=0, b=0,ref_Length=1, layer = 0):
    D = Device('MZI 1by1')
    L_MMI=D.add_ref(Beam_Splitter_MMI12(WGwidth = WGwidth, MMI_coupling_length=MMI_coupling_length, MMI_coupling_width = MMI_coupling_width, port_gap34 = port_gap34, buffer_length1 = buffer_length1,buffer_length = buffer_length, a=a, b=b))
    R_MMI=D.add_ref(Beam_Splitter_MMI12(WGwidth = WGwidth, MMI_coupling_length=MMI_coupling_length, MMI_coupling_width = MMI_coupling_width, port_gap34 = port_gap34, buffer_length1 = buffer_length1,buffer_length = buffer_length, a=a, b=b))
    # R_MMI.232W((3,10))
    U_wg=D.add_ref(waveguide(width=WGwidth,length=common_length))
    L_wg=D.add_ref(waveguide(width=WGwidth,length=common_length))

    U_wg.connect(port = 'wgport1', destination = L_MMI.ports['wgport3'])
    L_wg.connect(port = 'wgport1', destination = L_MMI.ports['wgport4'])
    R_MMI.connect(port = 'wgport3', destination = U_wg.ports['wgport2'])

    
    length_differance = 2*L_MMI.info['length_differance']

    D.add_port(name = 'wgport1', midpoint = L_MMI.ports['wgport1'].midpoint, width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = R_MMI.ports['wgport1'].midpoint, width = WGwidth, orientation =0)
    D.info['length_differance'] = length_differance
    
    return D 

def Race_track_pulley(WGwidth = 1, coupling_gap = 1, coupling_angle=30, radius=100,race_width=1, race_length=20,taper_length=0, taper_width=3,Bus_buffer=10, electrode_gap=5,electrode_width=5,layer = 0):
    D = Device('MZI 1by1')
    LH_ring =D.add_ref(ccw_bend(radius = radius, width = race_width,theta=180))
    RH_ring =D.add_ref(cw_bend(radius = radius, width = race_width,theta=180))
    U_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
    L_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
    Bus_ring =D.add_ref(ccw_bend(radius = radius+coupling_gap+WGwidth/2+race_width/2, width = WGwidth, start_angle =180-coupling_angle/2,theta=coupling_angle))
    Bus_ring.x =Bus_ring.x
    UBus_ring =D.add_ref(ccw_bend(radius = radius+coupling_gap+WGwidth/2+race_width/2, width = WGwidth, start_angle =180-coupling_angle/2,theta=coupling_angle/2))
    LBus_ring =D.add_ref(cw_bend(radius = radius+coupling_gap+WGwidth/2+race_width/2, width = WGwidth, start_angle =180+coupling_angle/2,theta=coupling_angle/2))
    UBus_wg=D.add_ref(wide_line(WGwidth = WGwidth, length=Bus_buffer, taper_width=WGwidth, taper_length = 0))
    LBus_wg=D.add_ref(wide_line(WGwidth = WGwidth, length=Bus_buffer, taper_width=WGwidth, taper_length = 0))
    LBus_port =D.add_ref(ccw_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port_ring =D.add_ref(ccw_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port=D.add_ref(wide_line(WGwidth = WGwidth, length=radius+2*taper_length+race_length ,taper_width=WGwidth, taper_length = 0))
     
    
    U_wg.connect(port = 'wgport1', destination = LH_ring.ports['wgport1'])
    L_wg.connect(port = 'wgport1', destination = LH_ring.ports['wgport2']) 
  
    U_Erode=D.add_ref(Electrode_xline(U_wg, gap=electrode_gap, width=electrode_width,layer=2))
    L_Erode=D.add_ref(Electrode_xline(L_wg, gap=electrode_gap, width=electrode_width,layer=2))
    
    RH_ring.connect(port = 'wgport1', destination = U_wg.ports['wgport2'])
    L_Erode=D.add_ref(Electrode_cwring(RH_ring, start_angle=30,theta=60,gap=electrode_gap, width=electrode_width,layer=2))
    
    UBus_ring.connect(port = 'wgport1', destination = Bus_ring.ports['wgport1'])
    LBus_ring.connect(port = 'wgport1', destination = Bus_ring.ports['wgport2'])
    UBus_wg.connect(port = 'wgport1', destination = UBus_ring.ports['wgport2'])
    LBus_wg.connect(port = 'wgport1', destination = LBus_ring.ports['wgport2']) 
    LBus_port.connect(port = 'wgport1', destination = UBus_wg.ports['wgport2'])
    RBus_port_ring.connect(port = 'wgport1', destination = LBus_wg.ports['wgport2']) 
    RBus_port.connect(port = 'wgport1', destination = RBus_port_ring.ports['wgport2'])
    D.add_port(name = 'wgport1', midpoint = LBus_port.ports['wgport2'].midpoint, width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = RBus_port.ports['wgport2'].midpoint, width = WGwidth, orientation =0)
  


    return D 


def Race_track(WGwidth = 1, coupling_gap = 1, coupling_angle=30, radius=100,race_width=1, race_length=20,taper_length=0, taper_width=3,Bus_buffer=100, layer = 0):
    D = Device('MZI 1by1')
    LH_ring =D.add_ref(ccw_bend(radius = radius, width = race_width,theta=180))
    RH_ring =D.add_ref(cw_bend(radius = radius, width = race_width,theta=180))
    U_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
    L_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
    Bus_wg=D.add_ref(wide_line(WGwidth = WGwidth, length=Bus_buffer, taper_width=WGwidth, taper_length = 0))
    Bus_wg.rotate(angle=90,center=(Bus_wg.x,Bus_wg.y))
    Bus_wg.x=-radius-coupling_gap-WGwidth/2-race_width/2
    LBus_port =D.add_ref(ccw_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port_ring =D.add_ref(ccw_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port=D.add_ref(wide_line(WGwidth = WGwidth, length=radius+2*taper_length+race_length ,taper_width=WGwidth, taper_length = 0))
     
    
    U_wg.connect(port = 'wgport1', destination = LH_ring.ports['wgport1'])
    L_wg.connect(port = 'wgport1', destination = LH_ring.ports['wgport2']) 
    RH_ring.connect(port = 'wgport1', destination = U_wg.ports['wgport2'])
    LBus_port.connect(port = 'wgport1', destination = Bus_wg.ports['wgport2'])
    RBus_port_ring.connect(port = 'wgport1', destination = Bus_wg.ports['wgport1']) 
    RBus_port.connect(port = 'wgport1', destination = RBus_port_ring.ports['wgport2'])
    D.add_port(name = 'wgport1', midpoint = LBus_port.ports['wgport2'].midpoint, width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = RBus_port.ports['wgport2'].midpoint, width = WGwidth, orientation =0)
  
    
    return D 

def Race_track_euler(WGwidth = 1, coupling_gap = 1, coupling_angle=30, radius=100,race_width=1, race_length=20,taper_length=0, taper_width=3,Bus_buffer=100, layer = 0):
    D = Device('MZI 1by1')
    RH_ring =D.add_ref(euler_bend(radius = radius, width = race_width,theta=180,p=0.2))
    LH_ring =D.add_ref(euler_bend(radius = radius, width = race_width,theta=-180,p=0.2))
    U_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
    L_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
  
    
    U_wg.connect(port = 'wgport1', destination = RH_ring.ports['wgport1'])
    L_wg.connect(port = 'wgport1', destination = RH_ring.ports['wgport2']) 
    LH_ring.connect(port = 'wgport1', destination = U_wg.ports['wgport2'])
    
    Bus_wg=D.add_ref(wide_line(WGwidth = WGwidth, length=Bus_buffer, taper_width=WGwidth, taper_length = 0))
    Bus_wg.rotate(angle=90,center=(Bus_wg.x,Bus_wg.y))
    Bus_wg.x=LH_ring.xmin-coupling_gap-WGwidth/2
    Bus_wg.y=LH_ring.y
    LBus_port =D.add_ref(euler_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port_ring =D.add_ref(euler_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port=D.add_ref(wide_line(WGwidth = WGwidth, length=radius+2*taper_length+race_length ,taper_width=WGwidth, taper_length = 0))
   
    
    LBus_port.connect(port = 'wgport1', destination = Bus_wg.ports['wgport2'])
    RBus_port_ring.connect(port = 'wgport1', destination = Bus_wg.ports['wgport1']) 
    RBus_port.connect(port = 'wgport1', destination = RBus_port_ring.ports['wgport2'])
    D.add_port(name = 'wgport1', midpoint = LBus_port.ports['wgport2'].midpoint, width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = RBus_port.ports['wgport2'].midpoint, width = WGwidth, orientation =0)
  
    
    return D

def Race_track_euler_pulley(WGwidth = 1, coupling_gap = 1, coupling_angle=1, radius=100,race_width=1, race_length=20,taper_length=0, taper_width=3,Bus_buffer=10, electrode_gap=5,electrode_width=5,layer = 0):
    D = Device('MZI 1by1')
    RH_ring =D.add_ref(euler_bend(radius = radius, width = race_width,theta=-180))
    LH_ring =D.add_ref(euler_bend(radius = radius, width = race_width,theta=180))
    U_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
    L_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
    
    U_wg.connect(port = 'wgport1', destination = RH_ring.ports['wgport1'])
    L_wg.connect(port = 'wgport1', destination = RH_ring.ports['wgport2']) 
    LH_ring.connect(port = 'wgport1', destination = U_wg.ports['wgport2'])
    
    Bus_ring =D.add_ref(ccw_bend(radius = radius+coupling_gap+WGwidth/2+race_width/2, width = WGwidth, start_angle =180-coupling_angle/2,theta=coupling_angle))
    Bus_ring.xmin =LH_ring.xmin-coupling_gap-WGwidth
    Bus_ring.y =LH_ring.y
    UBus_ring =D.add_ref(euler_bend(radius = radius+coupling_gap+WGwidth/2+race_width/2, width = WGwidth, theta=coupling_angle/2))
    LBus_ring =D.add_ref(euler_bend(radius = radius+coupling_gap+WGwidth/2+race_width/2, width = WGwidth, theta=-coupling_angle/2))
    UBus_wg=D.add_ref(wide_line(WGwidth = WGwidth, length=Bus_buffer, taper_width=WGwidth, taper_length = 0))
    LBus_wg=D.add_ref(wide_line(WGwidth = WGwidth, length=Bus_buffer, taper_width=WGwidth, taper_length = 0))
    LBus_port =D.add_ref(euler_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port_ring =D.add_ref(euler_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port=D.add_ref(wide_line(WGwidth = WGwidth, length=radius+2*taper_length+race_length ,taper_width=WGwidth, taper_length = 0))
     

    U_Erode=D.add_ref(Electrode_xline(U_wg, gap=electrode_gap/2, width=electrode_width,layer=2))
    L_Erode=D.add_ref(Electrode_xline(L_wg, gap=electrode_gap/2, width=electrode_width,layer=2))
    

    L_Erode=D.add_ref(Electrode_cwring(RH_ring, start_angle=30,theta=60,gap=electrode_gap/2, width=electrode_width,layer=2))
    L_Erode.xmax=RH_ring.xmax+electrode_gap/2+electrode_width
    L_Erode.y=RH_ring.y
    UBus_ring.connect(port = 'wgport1', destination = Bus_ring.ports['wgport1'])
    LBus_ring.connect(port = 'wgport1', destination = Bus_ring.ports['wgport2'])
    UBus_wg.connect(port = 'wgport1', destination = UBus_ring.ports['wgport2'])
    LBus_wg.connect(port = 'wgport1', destination = LBus_ring.ports['wgport2']) 
    LBus_port.connect(port = 'wgport1', destination = UBus_wg.ports['wgport2'])
    RBus_port_ring.connect(port = 'wgport1', destination = LBus_wg.ports['wgport2']) 
    RBus_port.connect(port = 'wgport1', destination = RBus_port_ring.ports['wgport2'])
    D.add_port(name = 'wgport1', midpoint = LBus_port.ports['wgport2'].midpoint, width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = RBus_port.ports['wgport2'].midpoint, width = WGwidth, orientation =0)
  


    return D 

def zRace_track_euler_pulley(WGwidth = 1, coupling_gap = 1, coupling_angle=1, radius=100,race_width=1, race_length=20,taper_length=0, taper_width=3,Bus_buffer=10, electrode_gap=5,electrode_width=5,layer = 0):
    D = Device('MZI 1by1')
    RH_ring =D.add_ref(euler_bend(radius = radius, width = race_width,theta=-180))
    LH_ring =D.add_ref(euler_bend(radius = radius, width = race_width,theta=180))
    U_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
    L_wg=D.add_ref(wide_line(WGwidth = race_width,length=race_length, taper_width=taper_width, taper_length = taper_length))
    

    U_wg.connect(port = 'wgport1', destination = RH_ring.ports['wgport1'])
    L_wg.connect(port = 'wgport1', destination = RH_ring.ports['wgport2']) 
    LH_ring.connect(port = 'wgport1', destination = U_wg.ports['wgport2'])
    
    Bus_ring =D.add_ref(ccw_bend(radius = radius+coupling_gap+WGwidth/2+race_width/2, width = WGwidth, start_angle =180-coupling_angle/2,theta=coupling_angle))
    Bus_ring.xmin =LH_ring.xmin-coupling_gap-WGwidth
    Bus_ring.y =LH_ring.y
    UBus_ring =D.add_ref(euler_bend(radius = radius+coupling_gap+WGwidth/2+race_width/2, width = WGwidth, theta=coupling_angle/2))
    LBus_ring =D.add_ref(euler_bend(radius = radius+coupling_gap+WGwidth/2+race_width/2, width = WGwidth, theta=-coupling_angle/2))
    UBus_wg=D.add_ref(wide_line(WGwidth = WGwidth, length=Bus_buffer+radius, taper_width=WGwidth, taper_length = 0))
    LBus_wg=D.add_ref(wide_line(WGwidth = WGwidth, length=Bus_buffer, taper_width=WGwidth, taper_length = 0))
    R2Bus_port =D.add_ref(euler_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port_ring =D.add_ref(euler_bend(radius = radius, width = WGwidth, theta=90))
    RBus_port=D.add_ref(wide_line(WGwidth = WGwidth, length=2*taper_length+race_length ,taper_width=WGwidth, taper_length = 0))
     

    U_Erode=D.add_ref(Electrode_xline(U_wg, gap=electrode_gap/2, width=electrode_width,layer=2))
    L_Erode=D.add_ref(Electrode_xline(L_wg, gap=electrode_gap/2, width=electrode_width,layer=2))
    

    L_Erode=D.add_ref(Electrode_cwring(RH_ring, start_angle=30,theta=60,gap=electrode_gap/2, width=electrode_width,layer=2))
    L_Erode.xmax=RH_ring.xmax+electrode_gap/2+electrode_width
    L_Erode.y=RH_ring.y
    UBus_ring.connect(port = 'wgport1', destination = Bus_ring.ports['wgport1'])
    LBus_ring.connect(port = 'wgport1', destination = Bus_ring.ports['wgport2'])
    UBus_wg.connect(port = 'wgport1', destination = UBus_ring.ports['wgport2'])
    LBus_wg.connect(port = 'wgport1', destination = LBus_ring.ports['wgport2']) 
    RBus_port_ring.connect(port = 'wgport1', destination = LBus_wg.ports['wgport2']) 
    RBus_port.connect(port = 'wgport1', destination = RBus_port_ring.ports['wgport2'])
    R2Bus_port.connect(port = 'wgport2', destination = RBus_port.ports['wgport2'])
    D.add_port(name = 'wgport1', midpoint = UBus_wg.ports['wgport2'].midpoint, width = WGwidth, orientation = 90)
    D.add_port(name = 'wgport2', midpoint = R2Bus_port.ports['wgport1'].midpoint, width = WGwidth, orientation =270)
    D.info['length']=RH_ring.info['length']+LH_ring.info['length']+U_wg.xsize+L_wg.xsize 


    return D 
def PSR_TEST(WGwidth = 0.6,Twidth1 = 1.5,Twidth2 = 2.5,Tlength1 = 1,Tlength2 = 2,Tlength3 = 1,outWGwidth1 = 1.1,outWGwidth2 = 1.4,port_angle=0.3,outlength = 2, layer = 0):
    D = Device('PSR_adiabaticity')
    Twidth3=outWGwidth1+outWGwidth2
    Input=D.add_ref(waveguide(WGwidth,1))
    Taper1 = D.add_ref(taper(WGwidth = WGwidth, min_width=Twidth1, taper_length = Tlength1))
    Taper2 = D.add_ref(taper(WGwidth = Twidth1, min_width=Twidth2, taper_length = Tlength2))
    Taper3 = D.add_ref(taper(WGwidth = Twidth2, min_width=Twidth3, taper_length = Tlength3))
    Taper1.connect(port = 'wgport1', destination = Input.ports['wgport2'])
    Taper2.connect(port = 'wgport1', destination = Taper1.ports['wgport2'])
    Taper3.connect(port = 'wgport1', destination = Taper2.ports['wgport2'])
    buffer1=D.add_ref(waveguide(outWGwidth1,1))
    buffer2=D.add_ref(waveguide(outWGwidth2,1))
    buffer1.connect(port = 'wgport1', destination = Taper3.ports['wgport2'])
    buffer1.y=buffer1.y-outWGwidth2/2
    buffer2.connect(port = 'wgport1', destination = Taper3.ports['wgport2'])
    buffer2.y=buffer2.y+outWGwidth1/2    
    
    outport1=D.add_ref(taper_port(WGwidth = WGwidth, WGlength=2,  taper_width=outWGwidth1, taper_length = 20, layer = 0))
    outport1.connect(port = 'wgport2', destination = buffer1.ports['wgport2'])
    outport1.x= outport1.x + outlength
    outport2=D.add_ref(taper_port(WGwidth = WGwidth, WGlength=2,  taper_width=outWGwidth2, taper_length = 20, layer = 0))
    outport2.connect(port = 'wgport2', destination = buffer2.ports['wgport2'])
    outport2.x = outport2.x + outlength
    outport2.y = outport2.y + outlength*tan(port_angle*pi/180)
    D.add_ref(pr.route_basic(port1 = outport1.ports['wgport2'], port2 = buffer1.ports['wgport2'], path_type = 'straight', width_type = 'sine'))
    D.add_ref(pr.route_basic(port1 = outport2.ports['wgport2'], port2 = buffer2.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
  
    D.add_port(name = 'wgport1', midpoint = Input.ports['wgport1'].midpoint, width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = outport2.ports['wgport1'].midpoint, width = WGwidth, orientation =0)
    D.add_port(name = 'wgport3', midpoint = outport1.ports['wgport1'].midpoint, width = WGwidth, orientation =0)

    return D 

def PSR_adiabaticity(WGwidth = 0.6,Twidth1 = 1.5,Twidth2 = 2.5,Tlength1 = 1,Tlength2 = 2,Tlength3 = 1,outWGwidth1 = 1.1,outWGwidth2 = 1.4,port_angle=0.3,outlength = 2, layer = 0):
    D = Device('PSR_adiabaticity')
    Twidth3=outWGwidth1+outWGwidth2
    Input=D.add_ref(waveguide(WGwidth,1))
    Taper1 = D.add_ref(taper(WGwidth = WGwidth, min_width=Twidth1, taper_length = Tlength1))
    Taper2 = D.add_ref(taper(WGwidth = Twidth1, min_width=Twidth2, taper_length = Tlength2))
    Taper3 = D.add_ref(taper(WGwidth = Twidth2, min_width=Twidth3, taper_length = Tlength3))
    Taper1.connect(port = 'wgport1', destination = Input.ports['wgport2'])
    Taper2.connect(port = 'wgport1', destination = Taper1.ports['wgport2'])
    Taper3.connect(port = 'wgport1', destination = Taper2.ports['wgport2'])
    buffer1=D.add_ref(waveguide(outWGwidth1,1))
    buffer2=D.add_ref(waveguide(outWGwidth2,1))
    buffer1.connect(port = 'wgport1', destination = Taper3.ports['wgport2'])
    buffer1.y=buffer1.y-outWGwidth2/2
    buffer2.connect(port = 'wgport1', destination = Taper3.ports['wgport2'])
    buffer2.y=buffer2.y+outWGwidth1/2    
    
    outport1=D.add_ref(taper_port(WGwidth = WGwidth, WGlength=2,  taper_width=outWGwidth1, taper_length = 20, layer = 0))
    outport1.connect(port = 'wgport2', destination = buffer1.ports['wgport2'])
    outport1.x= outport1.x + outlength
    outport2=D.add_ref(taper_port(WGwidth = WGwidth, WGlength=2,  taper_width=outWGwidth2, taper_length = 20, layer = 0))
    outport2.connect(port = 'wgport2', destination = buffer2.ports['wgport2'])
    outport2.x = outport2.x + outlength
    outport2.y = outport2.y + outlength*tan(port_angle*pi/180)
    D.add_ref(pr.route_basic(port1 = outport1.ports['wgport2'], port2 = buffer1.ports['wgport2'], path_type = 'straight', width_type = 'sine'))
    D.add_ref(pr.route_basic(port1 = outport2.ports['wgport2'], port2 = buffer2.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
  
    D.add_port(name = 'wgport1', midpoint = Input.ports['wgport1'].midpoint, width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = outport2.ports['wgport1'].midpoint, width = WGwidth, orientation =0)
    D.add_port(name = 'wgport3', midpoint = outport1.ports['wgport1'].midpoint, width = WGwidth, orientation =0)

    return D 
def Beam_Splitter_sym_MMI42(WGwidth = 1, taper_length=10, taper_width=1.2,radius=200, MMI_coupling_length=100, MMI_coupling_width = 0.5,coupling_port_gap=0.5, in_port_gap = 301.1, out_port_gap = 301.1, in_buffer_length = 10, out_buffer_length = 10, a=0, b=0,ref_Length=1, layer = 0):
    D = Device('Beam spliter with 4by2')

    wg_L_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port1.y = in_port_gap*3/2
    wg_L_port1.xmin = 0
    wg_L_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port2.y = in_port_gap/2
    wg_L_port2.xmin = 0
    wg_L_port3 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port3.y = -in_port_gap/2
    wg_L_port3.xmin = 0
    wg_L_port4 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port4.y = -in_port_gap*3/2
    wg_L_port4.xmin = 0

    MMI1 = D.add_ref(MMI22(length=MMI_coupling_length, width = MMI_coupling_width, taper_length=taper_length,port_gap_ratio=coupling_port_gap, taper_width=taper_width,port_width=WGwidth)) #coupling region 
    MMI1.y= 0
    MMI1.xmin = wg_L_port1.xmax+in_buffer_length
    
    
    R1=D.add_ref( pr.route_basic(port1 = MMI1.ports['wgport1'], port2 = wg_L_port2.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    R2=D.add_ref( pr.route_basic(port1 = MMI1.ports['wgport2'], port2 = wg_L_port3.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    R1u=D.add_ref( pr.route_basic(port1 = MMI1.ports['wgport1'], port2 = wg_L_port2.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    R1u.y=R1u.y+in_port_gap
    R2l=D.add_ref( pr.route_basic(port1 = MMI1.ports['wgport2'], port2 = wg_L_port3.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    R2l.y=R2l.y-in_port_gap
    
    MMI1u = D.add_ref(MMI11(length=MMI_coupling_length, width = taper_width, taper_length=taper_length, taper_width=taper_width,port_width=WGwidth)) #coupling region 
    MMI1u.xmin = wg_L_port1.xmax+in_buffer_length
    MMI1u.y = in_port_gap+MMI_coupling_width/4
    MMI1d = D.add_ref(MMI11(length=MMI_coupling_length, width = taper_width, taper_length=taper_length, taper_width=taper_width,port_width=WGwidth)) #coupling region 
    MMI1d.y= -in_port_gap-MMI_coupling_width/4
    MMI1d.xmin = wg_L_port1.xmax+in_buffer_length
    MMI2_1 = D.add_ref(MMI22(length=MMI_coupling_length, width = MMI_coupling_width, taper_length=taper_length,port_gap_ratio=coupling_port_gap, taper_width=taper_width,port_width=WGwidth)) #coupling region 
    MMI2_1.y= in_port_gap/2
    MMI2_1.xmin = wg_L_port1.xmax+in_buffer_length*2+MMI1.xsize
    MMI2_2 = D.add_ref(MMI22(length=MMI_coupling_length, width = MMI_coupling_width, taper_length=taper_length,port_gap_ratio=coupling_port_gap, taper_width=taper_width,port_width=WGwidth)) #coupling region 
    MMI2_2.y= -in_port_gap/2
    MMI2_2.xmin = wg_L_port1.xmax+in_buffer_length*2+MMI1.xsize
    
    

    bend0=D.add_ref(euler_bend(radius=radius,width=WGwidth,theta=180))
    bend0.connect(port='wgport1',destination=MMI2_1.ports['wgport3']) 
    bend1=D.add_ref(euler_bend(radius=radius,width=WGwidth,theta=-180))
    bend1.connect(port='wgport1',destination=MMI2_2.ports['wgport4']) 
    wg_R_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
    wg_R_port1.y = b - out_port_gap/2 +in_port_gap/2     
    wg_R_port1.xmin = wg_L_port1.xsize+in_buffer_length*2+MMI1.xsize*2+out_buffer_length   
    wg_R_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
    wg_R_port2.y = -b + out_port_gap/2 -in_port_gap/2     
    wg_R_port2.xmin = wg_L_port1.xsize+in_buffer_length*2+MMI1.xsize*2+out_buffer_length  
    wg_L_port1a = D.add_ref(taper_port(WGwidth = 0.1, WGlength=1,  taper_width=WGwidth, taper_length = 20))  #left port
    wg_L_port1a.connect(port='wgport2', destination=bend0.ports['wgport2'])
    wg_L_port2a = D.add_ref(taper_port(WGwidth = 0.1, WGlength=1,  taper_width=WGwidth, taper_length = 20))  #left port
    wg_L_port2a.connect(port='wgport2', destination=bend1.ports['wgport2'])
    
    
    
    
    #D.add_ref( pr.route_basic(port1 = MMI1u.ports['wgport1'], port2 = wg_L_port1.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    #D.add_ref( pr.route_basic(port1 = MMI1d.ports['wgport1'], port2 = wg_L_port4.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  

    D.add_ref( pr.route_basic(port1 = MMI2_1.ports['wgport1'], port2 = MMI1u.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    D.add_ref( pr.route_basic(port1 = MMI2_1.ports['wgport2'], port2 = MMI1.ports['wgport3'], path_type = 'sine', width_type = 'sine'))  
    D.add_ref( pr.route_basic(port1 = MMI2_2.ports['wgport1'], port2 = MMI1.ports['wgport4'], path_type = 'sine', width_type = 'sine'))  
    D.add_ref( pr.route_basic(port1 = MMI2_2.ports['wgport2'], port2 = MMI1d.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  

    D.add_ref( pr.route_basic(port1 = wg_R_port1.ports['wgport1'], port2 = MMI2_1.ports['wgport4'], path_type = 'sine', width_type = 'sine'))
    D.add_ref( pr.route_basic(port1 = wg_R_port2.ports['wgport1'], port2 = MMI2_2.ports['wgport3'], path_type = 'sine', width_type = 'sine'))
 
#    length_differance = er1.info['length']-er2.info['length']
    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.xmin,wg_L_port1.y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_L_port2.xmin,wg_L_port2.y], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport3', midpoint = [wg_L_port3.xmin,wg_L_port3.y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport4', midpoint = [wg_L_port4.xmin,wg_L_port4.y], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport5', midpoint = [wg_R_port1.xmax,wg_R_port1.y], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport6', midpoint = [wg_R_port2.xmax,wg_R_port2.y], width = WGwidth, orientation =0)

#    D.info['length_differance'] = length_differance

    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D 

def Switch_14(WGwidth = 1, taper_length=10, taper_width=1.2,EO_length=5000, MMI12_coupling_length=8, MMI12_coupling_width = 3, MMI22_coupling_length=56, MMI22_coupling_width = 5,coupling_port_gap=0.5, in_port_gap = 301.1, out_port_gap = 301.1, buffer_length = 1000, buffer_length2 = 1000, a=0, b=0,ref_Length=1, layer = 0,E_padx=60,E_pady=60,E_gap=5):
    D = Device('Optical switch 1by4')

    wg_L_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port1.y = 0
    wg_L_port1.xmin = 0

    MMI1_1 = D.add_ref(Beam_Splitter_MMI12(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI12_coupling_length, MMI_coupling_width = MMI12_coupling_width,coupling_port_gap=coupling_port_gap, port_gap34 = in_port_gap, buffer_length1 = 1, buffer_length = buffer_length, a=0, b=0,ref_Length=1, layer = 0)) #coupling region 
    MMI1_1.y= 0
    MMI1_1.xmin = wg_L_port1.xmax+buffer_length
    modul1_0=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul1_0.connect(port='wgport1',destination=MMI1_1.ports['wgport3'])
    modul1_1=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul1_1.connect(port='wgport1',destination=MMI1_1.ports['wgport4'])
    MMI1_2 = D.add_ref(Beam_Splitter_MMI22(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI22_coupling_length, MMI_coupling_width = MMI22_coupling_width,coupling_port_gap=coupling_port_gap, port_gap12 = in_port_gap, port_gap34 = out_port_gap*2, buffer_length = buffer_length, buffer_length2 = buffer_length*2, a=a, b=b,ref_Length=ref_Length, layer = 0)) #coupling region 
    MMI1_2.connect(port='wgport1',destination=modul1_0.ports['wgport2'])

    MMI2_1 = D.add_ref(Beam_Splitter_MMI12(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI12_coupling_length, MMI_coupling_width = MMI12_coupling_width,coupling_port_gap=coupling_port_gap, port_gap34 = in_port_gap, buffer_length1 = 1, buffer_length = buffer_length, a=0, b=0,ref_Length=1, layer = 0)) #coupling region
    MMI2_1.connect(port='wgport1',destination=MMI1_2.ports['wgport3'])
    modul2_0=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul2_0.connect(port='wgport1',destination=MMI2_1.ports['wgport3'])
    modul2_1=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul2_1.connect(port='wgport1',destination=MMI2_1.ports['wgport4'])
    MMI2_2 = D.add_ref(Beam_Splitter_MMI22(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI22_coupling_length, MMI_coupling_width = MMI22_coupling_width,coupling_port_gap=coupling_port_gap, port_gap12 = in_port_gap, port_gap34 = out_port_gap, buffer_length = buffer_length, buffer_length2 = buffer_length2, a=a, b=b,ref_Length=ref_Length, layer = 0)) #coupling region 
    MMI2_2.connect(port='wgport1',destination=modul2_0.ports['wgport2'])

    MMI3_1 = D.add_ref(Beam_Splitter_MMI12(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI12_coupling_length, MMI_coupling_width = MMI12_coupling_width,coupling_port_gap=coupling_port_gap, port_gap34 = in_port_gap, buffer_length1 = 1, buffer_length = buffer_length, a=0, b=0,ref_Length=1, layer = 0)) #coupling region
    MMI3_1.connect(port='wgport1',destination=MMI1_2.ports['wgport4'])
    modul3_0=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul3_0.connect(port='wgport1',destination=MMI3_1.ports['wgport3'])
    modul3_1=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul3_1.connect(port='wgport1',destination=MMI3_1.ports['wgport4'])
    MMI3_2 = D.add_ref(Beam_Splitter_MMI22(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI22_coupling_length, MMI_coupling_width = MMI22_coupling_width,coupling_port_gap=coupling_port_gap, port_gap12 = in_port_gap, port_gap34 = out_port_gap, buffer_length = buffer_length, buffer_length2 = buffer_length2, a=a, b=b,ref_Length=ref_Length, layer = 0)) #coupling region 
    MMI3_2.connect(port='wgport1',destination=modul3_0.ports['wgport2'])
    
    
#    length_differance = er1.info['length']-er2.info['length']
    D.add_port(name = 'wgport1', midpoint = [MMI1_1.ports['wgport1'].x,MMI1_1.ports['wgport1'].y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [MMI2_2.ports['wgport3'].x,MMI2_2.ports['wgport3'].y], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport3', midpoint = [MMI2_2.ports['wgport4'].x,MMI2_2.ports['wgport4'].y], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport4', midpoint = [MMI3_2.ports['wgport3'].x,MMI3_2.ports['wgport3'].y], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport5', midpoint = [MMI3_2.ports['wgport4'].x,MMI3_2.ports['wgport4'].y], width = WGwidth, orientation =0)

#    D.info['length_differance'] = length_differance

    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D


def CPC(WGwidth = 1, taper_length=10, taper_width=1.2,EO_length=5000, MMI12_coupling_length=8, MMI12_coupling_width = 3, MMI22_coupling_length=56, MMI22_coupling_width = 5,coupling_port_gap=0.5, in_port_gap = 301.1, out_port_gap = 301.1, buffer_length = 1000, buffer_length2 = 1000, a=0, b=0,ref_Length=1, layer = 0,E_padx=60,E_pady=60,E_gap=5):
    D = Device('polarization control')

    wg_L_port1 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port1.y = in_port_gap/2
    wg_L_port1.xmin = 0
    wg_L_port2 = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #left port
    wg_L_port2.y = -in_port_gap/2
    wg_L_port2.xmin = 0

    MMI1_1 = D.add_ref(Beam_Splitter_MMI22(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI22_coupling_length, MMI_coupling_width = MMI22_coupling_width,coupling_port_gap=coupling_port_gap, port_gap12 = in_port_gap, port_gap34 = out_port_gap, buffer_length = buffer_length, buffer_length2 = buffer_length, a=a, b=b,ref_Length=ref_Length, layer = 0))
    MMI1_1.xmin = wg_L_port1.xmax+buffer_length
    MMI1_1.connect(port='wgport1',destination=wg_L_port1.ports['wgport2'])
    
    modul1_0=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul1_0.connect(port='wgport1',destination=MMI1_1.ports['wgport3'])
    modul1_1=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul1_1.connect(port='wgport1',destination=MMI1_1.ports['wgport4'])
    MMI1_2 = D.add_ref(Beam_Splitter_MMI22(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI22_coupling_length, MMI_coupling_width = MMI22_coupling_width,coupling_port_gap=coupling_port_gap, port_gap12 = in_port_gap, port_gap34 = out_port_gap, buffer_length = buffer_length, buffer_length2 = buffer_length, a=a, b=b,ref_Length=ref_Length, layer = 0)) #coupling region 
    MMI1_2.connect(port='wgport1',destination=modul1_0.ports['wgport2'])

    # MMI2_1 = D.add_ref(Beam_Splitter_MMI12(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI12_coupling_length, MMI_coupling_width = MMI12_coupling_width,coupling_port_gap=coupling_port_gap, port_gap34 = in_port_gap, buffer_length1 = 1, buffer_length = buffer_length, a=0, b=0,ref_Length=1, layer = 0)) #coupling region
    # MMI2_1.connect(port='wgport1',destination=MMI1_2.ports['wgport3'])
    modul2_0=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul2_0.connect(port='wgport1',destination=MMI1_2.ports['wgport3'])
    modul2_1=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul2_1.connect(port='wgport1',destination=MMI1_2.ports['wgport4'])
    # MMI2_2 = D.add_ref(Beam_Splitter_MMI22(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI22_coupling_length, MMI_coupling_width = MMI22_coupling_width,coupling_port_gap=coupling_port_gap, port_gap12 = in_port_gap, port_gap34 = out_port_gap, buffer_length = buffer_length, buffer_length2 = buffer_length2, a=a, b=b,ref_Length=ref_Length, layer = 0)) #coupling region 
    # MMI2_2.connect(port='wgport1',destination=modul2_0.ports['wgport2'])

    # MMI3_1 = D.add_ref(Beam_Splitter_MMI12(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI12_coupling_length, MMI_coupling_width = MMI12_coupling_width,coupling_port_gap=coupling_port_gap, port_gap34 = in_port_gap, buffer_length1 = 1, buffer_length = buffer_length, a=0, b=0,ref_Length=1, layer = 0)) #coupling region
    # MMI3_1.connect(port='wgport1',destination=MMI1_2.ports['wgport4'])
    # modul3_0=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    # modul3_0.connect(port='wgport1',destination=MMI3_1.ports['wgport3'])
    # modul3_1=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    # modul3_1.connect(port='wgport1',destination=MMI3_1.ports['wgport4'])
    # MMI3_2 = D.add_ref(Beam_Splitter_MMI22(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI22_coupling_length, MMI_coupling_width = MMI22_coupling_width,coupling_port_gap=coupling_port_gap, port_gap12 = in_port_gap, port_gap34 = out_port_gap, buffer_length = buffer_length, buffer_length2 = buffer_length2, a=a, b=b,ref_Length=ref_Length, layer = 0)) #coupling region 
    # MMI3_2.connect(port='wgport1',destination=modul3_0.ports['wgport2'])
    
    
#    length_differance = er1.info['length']-er2.info['length']
    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.ports['wgport1'].x,wg_L_port1.ports['wgport1'].y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_L_port2.ports['wgport2'].x,wg_L_port2.ports['wgport2'].y], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport3', midpoint = [modul2_0.ports['wgport2'].x,modul2_0.ports['wgport2'].y], width = WGwidth, orientation =0)
    D.add_port(name = 'wgport4', midpoint = [modul2_1.ports['wgport2'].x,modul2_1.ports['wgport2'].y], width = WGwidth, orientation =0)
    # D.add_port(name = 'wgport5', midpoint = [MMI3_2.ports['wgport4'].x,MMI3_2.ports['wgport4'].y], width = WGwidth, orientation =0)

#    D.info['length_differance'] = length_differance

    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D

def Attenuator_11(WGwidth = 1, taper_length=10, taper_width=1.2,EO_length=5000, MMI12_coupling_length=8, MMI12_coupling_width = 3,coupling_port_gap=0.5, in_port_gap = 301.1, buffer_length = 1000, buffer_length2 = 1000, a=0, b=0,ref_Length=1, layer = 0,E_padx=60,E_pady=60,E_gap=5):
    D = Device('Optical switch 1by4')


    MMI1_1 = D.add_ref(Beam_Splitter_MMI12(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI12_coupling_length, MMI_coupling_width = MMI12_coupling_width,coupling_port_gap=coupling_port_gap, port_gap34 = in_port_gap, buffer_length1 = 1, buffer_length = buffer_length, a=a, b=b,ref_Length=ref_Length, layer = 0)) #coupling region 
    MMI1_1.y= 0
    MMI1_1.xmin = 0
    modul1_0=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul1_0.connect(port='wgport1',destination=MMI1_1.ports['wgport3'])
    modul1_1=D.add_ref(Elec_waveguide(width=WGwidth,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul1_1.connect(port='wgport1',destination=MMI1_1.ports['wgport4'])
    MMI1_2 = D.add_ref(Beam_Splitter_MMI12(WGwidth = WGwidth, taper_length=taper_length, taper_width=taper_width, MMI_coupling_length=MMI12_coupling_length, MMI_coupling_width = MMI12_coupling_width,coupling_port_gap=coupling_port_gap, port_gap34 = in_port_gap, buffer_length1 = 1, buffer_length = buffer_length, a=a, b=-b,ref_Length=ref_Length, layer = 0)) #coupling region 
    MMI1_2.rotate(180,(0,0))
    MMI1_2.connect(port='wgport4',destination=modul1_0.ports['wgport2'])

    
    
#    length_differance = er1.info['length']-er2.info['length']
    D.add_port(name = 'wgport1', midpoint = [MMI1_1.ports['wgport1'].x,MMI1_1.ports['wgport1'].y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [MMI1_2.ports['wgport1'].x,MMI1_2.ports['wgport1'].y], width = WGwidth, orientation =0)

#    D.info['length_differance'] = length_differance

    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D


def Pol_modul(WGwidth = 1, taper_length=10, taper_width=1.2,EO_length=5000, buffer_length = 1000, layer = 0,E_padx=60,E_pady=60,E_gap=3):
    D = Device('Optical switch 1by4')

    wg_L_port1 = D.add_ref(taper_port(WGwidth = WGwidth, WGlength=buffer_length,  taper_width=taper_width, taper_length = taper_length))  #left port
    wg_L_port1.y = 0
    wg_L_port1.xmin = 0

    modul0=D.add_ref(Elec_waveguide(width=taper_width,length=EO_length,E_padx=E_padx,E_pady=E_pady,E_gap=E_gap))
    modul0.connect(port='wgport1',destination=wg_L_port1.ports['wgport2'])

    wg_R_port1 = D.add_ref(taper_port(WGwidth = WGwidth, WGlength=buffer_length,  taper_width=taper_width, taper_length = taper_length))  #left port
    wg_R_port1.rotate(180,(0,0))
    wg_R_port1.connect(port='wgport2',destination=modul0.ports['wgport2'])

#    length_differance = er1.info['length']-er2.info['length']
    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.ports['wgport1'].x,wg_L_port1.ports['wgport1'].y], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_R_port1.ports['wgport1'].x,wg_R_port1.ports['wgport1'].y], width = WGwidth, orientation =0)

#    D.info['length_differance'] = length_differance

    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D

def PSR_adiabatic(WGwidth = 1, WGport1s= 2.26, WGport1m1= 2.78, WGport1m2= 2.98, WGport1e= 3.5,WGport2s= 1.57, WGport2m1= 1.31, WGport2m2= 1.21 ,WGport2e= 0.95,WGT1=2,WGT2=1.6,conv_length=500,coupling_length=1600, coupling_gap = 0.6, port_gap12 = 301.1, Taper_buffer_length = 100,Taper_buffer_length2=100,Taper_buffer_length3=450, buffer_length = 1000, a=0,ref_Length=1, layer = 0):
    D = Device('PSR with Adiabatic couple')
    wg_L_port1 = D.add_ref(taper_port2(WGleft = WGwidth, WGleftlength=ref_Length,  WGright=WGport1s,WGrightlength=ref_Length, taper_length = Taper_buffer_length))  #left port
    wg_L_port1.y = port_gap12/2
    wg_L_port1.xmin = 0
    wg_L_port2 = D.add_ref(taper_port2(WGleft = WGwidth, WGleftlength=ref_Length,  WGright=WGport2s,WGrightlength=ref_Length, taper_length = Taper_buffer_length))  #left port
    wg_L_port2.y = -port_gap12/2
    wg_L_port2.xmin = 0

    wg_CL1_1 = D.add_ref(taper(WGwidth = WGport1s, min_width=WGport1m1, taper_length = Taper_buffer_length3,shift=-(WGport1s-WGport1m1))) #coupling region 
    wg_CL1_1.ymin= a + (coupling_gap)/2
    wg_CL1_1.xmin = wg_L_port1.xsize+buffer_length
    wg_CL1_2 = D.add_ref(taper(WGwidth = WGport1m1, min_width=WGport1m2, taper_length = coupling_length,shift=-(WGport1m1-WGport1m2))) #coupling region 
    wg_CL1_2.connect(port='wgport1',destination=wg_CL1_1.ports['wgport2'])
    wg_CL1_3 = D.add_ref(taper(WGwidth = WGport1m2, min_width=WGport1e, taper_length = Taper_buffer_length3,shift=-(WGport1m2-WGport1e))) #coupling region 
    wg_CL1_3.connect(port='wgport1',destination=wg_CL1_2.ports['wgport2'])
 
    wg_CL2_1 = D.add_ref(taper(WGwidth = WGport2s, min_width=WGport2m1, taper_length = Taper_buffer_length3,shift=(WGport2s-WGport2m1))) #coupling region 
    wg_CL2_1.ymax= a - (coupling_gap)/2
    wg_CL2_1.xmin = wg_L_port1.xsize+buffer_length
    wg_CL2_2 = D.add_ref(taper(WGwidth = WGport2m1, min_width=WGport2m2, taper_length = coupling_length,shift=(WGport2m1-WGport2m2))) #coupling region 
    wg_CL2_2.connect(port='wgport1',destination=wg_CL2_1.ports['wgport2'])
    wg_CL2_3 = D.add_ref(taper(WGwidth = WGport2m2, min_width=WGport2e, taper_length = Taper_buffer_length3,shift=(WGport2m2-WGport2e))) #coupling region 
    wg_CL2_3.connect(port='wgport1',destination=wg_CL2_2.ports['wgport2'])

    wg_conv = D.add_ref(taper(WGwidth = WGT1, min_width=WGT2, taper_length = conv_length))  #TE1 to TM0      
    wg_conv.connect(port='wgport1',destination=wg_CL1_3.ports['wgport2'])
    wg_conv.xmin = wg_conv.xmin+ Taper_buffer_length2  
    
    wg_R_port = D.add_ref(waveguide(length = ref_Length, width=WGwidth))  #right port    
    wg_R_port.connect(port='wgport1',destination=wg_conv.ports['wgport2'])    
    wg_R_port.xmin = wg_R_port.xmin+Taper_buffer_length2  
    D.add_ref( pr.route_basic(port1 = wg_CL1_1.ports['wgport1'], port2 = wg_L_port1.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    D.add_ref( pr.route_basic(port1 = wg_CL2_1.ports['wgport1'], port2 = wg_L_port2.ports['wgport2'], path_type = 'sine', width_type = 'sine'))  
    D.add_ref( pr.route_basic(port1 = wg_CL1_3.ports['wgport2'], port2 = wg_conv.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    D.add_ref( pr.route_basic(port1 = wg_R_port.ports['wgport1'], port2 = wg_conv.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    D.add_port(name = 'wgport1', midpoint = [wg_L_port1.xmin,port_gap12/2], width = WGwidth, orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [wg_L_port2.xmin,-port_gap12/2], width = WGwidth, orientation =180)
    D.add_port(name = 'wgport3', midpoint = [wg_R_port.xmax,wg_R_port.ports['wgport2'].y], width = WGwidth, orientation =0)
    #가운데 웨이브가이드 설계-------------------------------------------------------------------------
    return D  


def Optimize_Adiabatic_DC(filepath ='C:/Users/kist/Documents/KIST 허형준/Fabrication/E-beam design/2023 EBL design/230421_adiabatic_DC/LN_design_code/Adia_DC.csv', width2=0.6, port_gap12 = 400, port_gap34 = 400, buffer_length = 1000, a=0, b=0,ref_Length=1, layer = 0):
    """ Creates an Adiabatic_DC"""
    df2 = pd.read_csv(filepath)
    xaxis=df2.values[:,0] 
    width_ans=df2.values[:,1] 
    gap_ans=df2.values[:,2] 
    t = np.linspace(0, xaxis[len(xaxis)-1], 1000)
    upper_points_x = (t).tolist()
    upper_points_y = (np.interp(t, xaxis,width_ans)+np.interp(t, xaxis,gap_ans)/2).tolist()
    lower_points_x = (t).tolist()
    lower_points_y = (np.interp(t, xaxis,gap_ans)/2).tolist()
    upper_points_x2 = (t).tolist()
    upper_points_y2 = (-np.interp(t, xaxis,gap_ans)/2).tolist()
    lower_points_x2 = (t).tolist()
    lower_points_y2 = (-np.interp(t, xaxis,gap_ans)/2-width2).tolist()
    xpts = upper_points_x + lower_points_x[::-1]
    ypts = upper_points_y + lower_points_y[::-1]
    xpts2 = upper_points_x2 + lower_points_x2[::-1]
    ypts2 = upper_points_y2 + lower_points_y2[::-1]

    D = Device('Adiabatic_DC')
    D.add_polygon(points = (xpts,ypts), layer = 0)  
    D.add_polygon(points = (xpts2,ypts2), layer = 0)     
    D.add_port(name = 'wgport1', midpoint = [xaxis[0],width_ans[0]/2+gap_ans[0]/2], width = width_ans[0], orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [xaxis[0],-width2/2-gap_ans[0]/2], width = width2, orientation = 180)
    D.add_port(name = 'wgport3', midpoint = [xaxis[len(xaxis)-1],width_ans[len(width_ans)-1]/2+gap_ans[len(gap_ans)-1]/2], width = width_ans[len(width_ans)-1], orientation = 0)
    D.add_port(name = 'wgport4', midpoint = [xaxis[len(xaxis)-1],-width2/2-gap_ans[len(gap_ans)-1]/2], width = width2, orientation = 0)
    D.info['length'] = xaxis[len(xaxis)-1]
#  creat port 
    D1 = Device('Add ports')
    O_A_DC = D1.add_ref(D)
    L_input1 = D1.add_ref(waveguide(length = ref_Length, width=width2))
    L_input1.y=port_gap12/2 
    L_input2 = D1.add_ref(waveguide(length = ref_Length, width=width2))
    L_input2.y=-port_gap12/2

    R_input1 = D1.add_ref(waveguide(length = ref_Length, width=width2))
    R_input2 = D1.add_ref(waveguide(length = ref_Length, width=width2))
    O_A_DC.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    O_A_DC.x = O_A_DC.x+buffer_length
    O_A_DC.y = O_A_DC.y -port_gap12/2+a
    R_input1.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    R_input1.x=R_input1.x+O_A_DC.xsize+buffer_length*2
    R_input2.y=R_input1.y-port_gap34-b 
    R_input2.x=R_input1.x
    D1.add_ref(pr.route_basic(port1 = L_input1.ports['wgport2'], port2 = O_A_DC.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = L_input2.ports['wgport2'], port2 = O_A_DC.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input1.ports['wgport1'], port2 = O_A_DC.ports['wgport3'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input2.ports['wgport1'], port2 = O_A_DC.ports['wgport4'], path_type = 'sine', width_type = 'sine'))
    
    D1.add_port(name = 'wgport1', midpoint = [L_input1.xmin,port_gap12/2], width = width2, orientation = 180)
    D1.add_port(name = 'wgport2', midpoint = [L_input1.xmin,-port_gap12/2], width = width2, orientation =180)
    D1.add_port(name = 'wgport3', midpoint = [R_input1.xmax,b + port_gap34/2], width = width2, orientation =0)
    D1.add_port(name = 'wgport4', midpoint = [R_input1.xmax,b - port_gap34/2], width = width2, orientation =0)

    return D1


def Rapid_Adiabatic_DC(filepath ='C:/Users/kist/Documents/KIST 허형준/Fabrication/E-beam design/2023 EBL design/230421_adiabatic_DC/LN_design_code/Adia_DC.csv', length = 100, gap = 0.6, port_gap12 = 400, port_gap34 = 100, width_ref = 0.6, height = 1.5, bend_length1 = 200, bend_length2 = 30, buffer_length = 1000, ref_Length=1):
    """ Creates an Rapid_Adiabatic_Coupler"""
    df2 = pd.read_csv(filepath, header=None)
    theta = df2.values[:,0]
    
    xaxis = np.linspace(0,length,30)
    width1_ans= np.linspace(0.5,0.65,len(xaxis))
    width2_ans= np.linspace(0.8,0.65,len(xaxis))
    t = xaxis
    bend1 = []
    bend2 = []
    bend3 = []
    bend4 = []
    
    upper_points_x = (t).tolist()
    upper_points_y = ( width1_ans + gap/2 + np.tan(theta)*length/len(xaxis) ).tolist()
    lower_points_x = (t).tolist()
    lower_points_y = (gap/2 + np.tan(theta)*length/len(xaxis)).tolist()
    upper_points_x2 = (t).tolist()
    upper_points_y2 = (-gap/2 + np.tan(theta)*length/len(xaxis)).tolist()
    lower_points_x2 = (t).tolist()
    lower_points_y2 = ( -width2_ans - gap/2 + np.tan(theta)*length/len(xaxis)).tolist()
    xpts = upper_points_x + lower_points_x[::-1]
    ypts = upper_points_y + lower_points_y[::-1]
    xpts2 = upper_points_x2 + lower_points_x2[::-1]
    ypts2 = upper_points_y2 + lower_points_y2[::-1]

    for  j in range(0, len(xaxis)+1 , 1):
        bend1.append((1-j/len(xaxis))**3*height + 3*j/len(xaxis)*(1-j/len(xaxis))**2*height)
        bend2.append((1-j/len(xaxis))**3*(-height) + 3*j/len(xaxis)*(1-j/len(xaxis))**2*(-height))
        bend3.append(3*(j/len(xaxis))**2*(1-j/len(xaxis))*height + (j/len(xaxis))**3*height)
        bend4.append(-3*(j/len(xaxis))**2*(1-j/len(xaxis))*height - (j/len(xaxis))**3*height)
    bend1 = np.array(bend1)
    bend2 = np.array(bend2)
    bend3 = np.array(bend3)
    bend4 = np.array(bend4)
    bend_x1 = (np.linspace(-bend_length1,0,len(xaxis)+1)).tolist()
    bend_x2 = (np.linspace(length,length+bend_length2,len(xaxis)+1)).tolist()
    
    upper_bend1_y = (width1_ans[0] + gap/2 + bend1 + np.tan(theta[0])*length/len(xaxis)).tolist()
    lower_bend1_y = (gap/2 + bend1 + np.tan(theta[0])*length/len(xaxis)).tolist()
    upper_bend2_y = (-gap/2 + bend2 + np.tan(theta[0])*length/len(xaxis)).tolist()
    lower_bend2_y = (-width2_ans[0] - gap/2 + bend2 + np.tan(theta[0])*length/len(xaxis)).tolist()
    upper_bend3_y = (width1_ans[-1] + gap/2 + bend3 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    lower_bend3_y = (gap/2 + bend3 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    upper_bend4_y = (-gap/2 + bend4 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    lower_bend4_y = (-width2_ans[-1] - gap/2 + bend4 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    bend_xpts1 = bend_x1 + bend_x1[::-1]
    bend_xpts2 = bend_x2 + bend_x2[::-1]
    bend_ypts1 = upper_bend1_y + lower_bend1_y[::-1]
    bend_ypts2 = upper_bend2_y + lower_bend2_y[::-1]
    bend_ypts3 = upper_bend3_y + lower_bend3_y[::-1]
    bend_ypts4 = upper_bend4_y + lower_bend4_y[::-1]
    
    D = Device('Rapid_Adiabatic_Coupler')
    D.add_polygon(points = (xpts,ypts), layer = 0)  
    D.add_polygon(points = (xpts2,ypts2), layer = 0)     
    D.add_polygon(points = (bend_xpts1, bend_ypts1), layer = 0)
    D.add_polygon(points = (bend_xpts1, bend_ypts2), layer = 0)
    D.add_polygon(points = (bend_xpts2, bend_ypts3), layer = 0)
    D.add_polygon(points = (bend_xpts2, bend_ypts4), layer = 0)
    
    D.add_port(name = 'wgport1', midpoint = [bend_x1[0], width1_ans[0]/2 + gap/2 + bend1[0] + np.tan(theta[0])*length/len(xaxis)], width = width1_ans[0], orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [bend_x1[0], -width2_ans[0]/2 - gap/2 + bend2[0] + np.tan(theta[0])*length/len(xaxis)], width = width2_ans[0], orientation = 180)
    D.add_port(name = 'wgport3', midpoint = [bend_x2[-1], width1_ans[-1]/2 + gap/2 + bend3[-1] + np.tan(theta[-1])*length/len(xaxis)], width = width1_ans[-1], orientation = 0)
    D.add_port(name = 'wgport4', midpoint = [bend_x2[-1], -width2_ans[-1]/2 - gap/2 + bend4[-1] + np.tan(theta[-1])*length/len(xaxis)], width = width2_ans[-1], orientation = 0)
    D.info['length'] = xaxis[len(xaxis)-1] + bend_length1 + bend_length2
    
#  creat port 
    D1 = Device('Add ports')
    R_A_DC = D1.add_ref(D)
    L_input1 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input1.y = port_gap12/2 
    L_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input2.y=-port_gap12/2

    R_input1 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    R_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    
    R_A_DC.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    R_A_DC.x = R_A_DC.x + buffer_length
    R_A_DC.y = R_A_DC.y -port_gap12/2
    R_input1.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    R_input1.x=R_input1.x+R_A_DC.xsize+buffer_length*2
    R_input1.y = port_gap34/2
    R_input2.x=R_input1.x
    R_input2.y=-port_gap34/2
    
    
    D1.add_ref(pr.route_basic(port1 = L_input1.ports['wgport2'], port2 = R_A_DC.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = L_input2.ports['wgport2'], port2 = R_A_DC.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input1.ports['wgport1'], port2 = R_A_DC.ports['wgport3'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input2.ports['wgport1'], port2 = R_A_DC.ports['wgport4'], path_type = 'sine', width_type = 'sine'))
    
    D1.add_port(name = 'wgport1', midpoint = [L_input1.xmin,port_gap12/2], width = width_ref, orientation = 180)
    D1.add_port(name = 'wgport2', midpoint = [L_input1.xmin,-port_gap12/2], width = width_ref, orientation =180)
    D1.add_port(name = 'wgport3', midpoint = [R_input1.xmax, port_gap34/2], width = width_ref, orientation =0)
    D1.add_port(name = 'wgport4', midpoint = [R_input1.xmax,-port_gap34/2], width = width_ref, orientation =0)
    
    return D1



def Rapid_Adiabatic_DC_v002(filepath ='C:/Users/kist/Documents/KIST 허형준/Fabrication/E-beam design/2023 EBL design/230421_adiabatic_DC/LN_design_code/Adia_DC.csv', length = 100, gap = 0.6, port_gap12 = 400, port_gap34 = 100, width_ref = 0.6, height = 1.5, bend_length1 = 200, bend_length2 = 30, buffer_length = 1000, ref_Length=1):
    """ Creates an Rapid_Adiabatic_Coupler"""
    df2 = pd.read_csv(filepath, header=None)
    theta = df2.values[:,0]
    
    xaxis = np.linspace(0,length,30)
    width1_ans= np.linspace(0.5,0.65,len(xaxis))
    width2_ans= np.linspace(0.8,0.65,len(xaxis))
    t = xaxis
    bend1 = []
    bend2 = []
    bend3 = []
    bend4 = []
    
    upper_points_x = (t).tolist()
    upper_points_y = ( width1_ans + gap/2 + np.tan(theta)*length/len(xaxis) ).tolist()
    lower_points_x = (t).tolist()
    lower_points_y = (gap/2 + np.tan(theta)*length/len(xaxis)).tolist()
    upper_points_x2 = (t).tolist()
    upper_points_y2 = (-gap/2 + np.tan(theta)*length/len(xaxis)).tolist()
    lower_points_x2 = (t).tolist()
    lower_points_y2 = ( -width2_ans - gap/2 + np.tan(theta)*length/len(xaxis)).tolist()
    xpts = upper_points_x + lower_points_x[::-1]
    ypts = upper_points_y + lower_points_y[::-1]
    xpts2 = upper_points_x2 + lower_points_x2[::-1]
    ypts2 = upper_points_y2 + lower_points_y2[::-1]

    for  j in range(0, len(xaxis)+1 , 1):
        bend1.append((1-j/len(xaxis))**3*height + 3*j/len(xaxis)*(1-j/len(xaxis))**2*height)
        bend2.append((1-j/len(xaxis))**3*(-height) + 3*j/len(xaxis)*(1-j/len(xaxis))**2*(-height))
        bend3.append(3*(j/len(xaxis))**2*(1-j/len(xaxis))*height + (j/len(xaxis))**3*height)
        bend4.append(-3*(j/len(xaxis))**2*(1-j/len(xaxis))*height - (j/len(xaxis))**3*height)
    bend1 = np.array(bend1)
    bend2 = np.array(bend2)
    bend3 = np.array(bend3)
    bend4 = np.array(bend4)
    bend_x1 = (np.linspace(-bend_length1,0,len(xaxis)+1)).tolist()
    bend_x2 = (np.linspace(length,length+bend_length2,len(xaxis)+1)).tolist()
    
    upper_bend1_y = (width1_ans[0] + gap/2 + bend1 + np.tan(theta[0])*length/len(xaxis)).tolist()
    lower_bend1_y = (gap/2 + bend1 + np.tan(theta[0])*length/len(xaxis)).tolist()
    upper_bend2_y = (-gap/2 + bend2 + np.tan(theta[0])*length/len(xaxis)).tolist()
    lower_bend2_y = (-width2_ans[0] - gap/2 + bend2 + np.tan(theta[0])*length/len(xaxis)).tolist()
    upper_bend3_y = (width1_ans[-1] + gap/2 + bend3 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    lower_bend3_y = (gap/2 + bend3 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    upper_bend4_y = (-gap/2 + bend4 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    lower_bend4_y = (-width2_ans[-1] - gap/2 + bend4 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    bend_xpts1 = bend_x1 + bend_x1[::-1]
    bend_xpts2 = bend_x2 + bend_x2[::-1]
    bend_ypts1 = upper_bend1_y + lower_bend1_y[::-1]
    bend_ypts2 = upper_bend2_y + lower_bend2_y[::-1]
    bend_ypts3 = upper_bend3_y + lower_bend3_y[::-1]
    bend_ypts4 = upper_bend4_y + lower_bend4_y[::-1]
    
    D = Device('Rapid_Adiabatic_Coupler')
    D.add_polygon(points = (xpts,ypts), layer = 0)  
    D.add_polygon(points = (xpts2,ypts2), layer = 0)     
    D.add_polygon(points = (bend_xpts1, bend_ypts1), layer = 0)
    D.add_polygon(points = (bend_xpts1, bend_ypts2), layer = 0)
    D.add_polygon(points = (bend_xpts2, bend_ypts3), layer = 0)
    D.add_polygon(points = (bend_xpts2, bend_ypts4), layer = 0)
    
    D.add_port(name = 'wgport1', midpoint = [bend_x1[0], width1_ans[0]/2 + gap/2 + bend1[0] + np.tan(theta[0])*length/len(xaxis)], width = width1_ans[0], orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [bend_x1[0], -width2_ans[0]/2 - gap/2 + bend2[0] + np.tan(theta[0])*length/len(xaxis)], width = width2_ans[0], orientation = 180)
    D.add_port(name = 'wgport3', midpoint = [bend_x2[-1], width1_ans[-1]/2 + gap/2 + bend3[-1] + np.tan(theta[-1])*length/len(xaxis)], width = width1_ans[-1], orientation = 0)
    D.add_port(name = 'wgport4', midpoint = [bend_x2[-1], -width2_ans[-1]/2 - gap/2 + bend4[-1] + np.tan(theta[-1])*length/len(xaxis)], width = width2_ans[-1], orientation = 0)
    D.info['length'] = xaxis[len(xaxis)-1] + bend_length1 + bend_length2
    
#  creat port 
    D1 = Device('Add ports')
    R_A_DC = D1.add_ref(D)
    L_input1 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input1.y = port_gap12/2 
    L_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input2.y=-port_gap12/2
    
    R_input1 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    R_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    

    # R_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    
    
    R_A_DC.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    R_A_DC.x = R_A_DC.x + buffer_length
    R_A_DC.y = R_A_DC.y -port_gap12/2
    R_input1.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    R_input1.x=R_input1.x+R_A_DC.xsize+buffer_length*2
    R_input1.y = port_gap34/2
    R_input2.x=R_input1.x
    R_input2.y=-port_gap34/2
    
    
    L_input1b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input1b.connect(port='wgport2', destination=R_A_DC.ports['wgport1'])
    L_input1b.move([-30,0])
    L_input2b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input2b.connect(port='wgport2', destination=R_A_DC.ports['wgport2'])
    L_input2b.move([-30,0])
    # L_input2.y=-port_gap12/2

    R_input1b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    R_input1b.connect(port='wgport1', destination=R_A_DC.ports['wgport3'])
    R_input1b.move([30,0])
    
    
    R_input2b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    R_input2b.connect(port='wgport1', destination=R_A_DC.ports['wgport4'])
    R_input2b.move([30,0])
    
    D1.add_ref(pr.route_basic(port1 = L_input1.ports['wgport2'], port2 = L_input1b.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = L_input2.ports['wgport2'], port2 = L_input2b.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input1.ports['wgport1'], port2 = R_input1b.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input2.ports['wgport1'], port2 = R_input2b.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    
    
    
    D1.add_ref(pr.route_basic(port1 = L_input1b.ports['wgport2'], port2 = R_A_DC.ports['wgport1'], path_type = 'straight', width_type = 'straight'))
    D1.add_ref(pr.route_basic(port1 = L_input2b.ports['wgport2'], port2 = R_A_DC.ports['wgport2'], path_type = 'straight', width_type = 'straight'))
    D1.add_ref(pr.route_basic(port1 = R_input1b.ports['wgport1'], port2 = R_A_DC.ports['wgport3'], path_type = 'straight', width_type = 'straight'))
    D1.add_ref(pr.route_basic(port1 = R_input2b.ports['wgport1'], port2 = R_A_DC.ports['wgport4'], path_type = 'straight', width_type = 'straight'))
    
    
    D1.add_port(name = 'wgport1', midpoint = [L_input1.xmin,port_gap12/2], width = width_ref, orientation = 180)
    D1.add_port(name = 'wgport2', midpoint = [L_input1.xmin,-port_gap12/2], width = width_ref, orientation =180)
    D1.add_port(name = 'wgport3', midpoint = [R_input1.xmax, port_gap34/2], width = width_ref, orientation =0)
    D1.add_port(name = 'wgport4', midpoint = [R_input1.xmax,-port_gap34/2], width = width_ref, orientation =0)
    
    return D1




def Rapid_Adiabatic_DC_v003(filepath ='C:/Users/kist/Documents/KIST 허형준/Fabrication/E-beam design/2023 EBL design/230421_adiabatic_DC/LN_design_code/Adia_DC.csv', length = 100, gap = 0.6, port_gap12 = 400, port_gap34 = 100, width_ref = 0.6, height = 1.5, bend_length1 = 200, bend_length2 = 30, buffer_length = 1000, ref_Length=1):
    """ Creates an Rapid_Adiabatic_Coupler"""
    df2 = pd.read_csv(filepath, header=None)
    theta = df2.values[:,0]
    
    xaxis = np.linspace(0,length,30)
    width1_ans= np.linspace(0.5,0.65,len(xaxis))
    width2_ans= np.linspace(0.8,0.65,len(xaxis))
    t = xaxis
    bend1 = []
    bend2 = []
    bend3 = []
    bend4 = []
    
    upper_points_x = (t).tolist()
    upper_points_y = ( width1_ans + gap/2 + np.tan(theta)*length/len(xaxis) ).tolist()
    lower_points_x = (t).tolist()
    lower_points_y = (gap/2 + np.tan(theta)*length/len(xaxis)).tolist()
    upper_points_x2 = (t).tolist()
    upper_points_y2 = (-gap/2 + np.tan(theta)*length/len(xaxis)).tolist()
    lower_points_x2 = (t).tolist()
    lower_points_y2 = ( -width2_ans - gap/2 + np.tan(theta)*length/len(xaxis)).tolist()
    xpts = upper_points_x + lower_points_x[::-1]
    ypts = upper_points_y + lower_points_y[::-1]
    xpts2 = upper_points_x2 + lower_points_x2[::-1]
    ypts2 = upper_points_y2 + lower_points_y2[::-1]

    for  j in range(0, len(xaxis)+1 , 1):
        bend1.append((1-j/len(xaxis))**3*height + 3*j/len(xaxis)*(1-j/len(xaxis))**2*height)
        bend2.append((1-j/len(xaxis))**3*(-height) + 3*j/len(xaxis)*(1-j/len(xaxis))**2*(-height))
        bend3.append(3*(j/len(xaxis))**2*(1-j/len(xaxis))*height + (j/len(xaxis))**3*height)
        bend4.append(-3*(j/len(xaxis))**2*(1-j/len(xaxis))*height - (j/len(xaxis))**3*height)
    bend1 = np.array(bend1)
    bend2 = np.array(bend2)
    bend3 = np.array(bend3)
    bend4 = np.array(bend4)
    bend_x1 = (np.linspace(-bend_length1,0,len(xaxis)+1)).tolist()
    bend_x2 = (np.linspace(length,length+bend_length2,len(xaxis)+1)).tolist()
    
    upper_bend1_y = (width1_ans[0] + gap/2 + bend1 + np.tan(theta[0])*length/len(xaxis)).tolist()
    lower_bend1_y = (gap/2 + bend1 + np.tan(theta[0])*length/len(xaxis)).tolist()
    upper_bend2_y = (-gap/2 + bend2 + np.tan(theta[0])*length/len(xaxis)).tolist()
    lower_bend2_y = (-width2_ans[0] - gap/2 + bend2 + np.tan(theta[0])*length/len(xaxis)).tolist()
    upper_bend3_y = (width1_ans[-1] + gap/2 + bend3 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    lower_bend3_y = (gap/2 + bend3 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    upper_bend4_y = (-gap/2 + bend4 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    lower_bend4_y = (-width2_ans[-1] - gap/2 + bend4 + np.tan(theta[-1])*length/len(xaxis)).tolist()
    bend_xpts1 = bend_x1 + bend_x1[::-1]
    bend_xpts2 = bend_x2 + bend_x2[::-1]
    bend_ypts1 = upper_bend1_y + lower_bend1_y[::-1]
    bend_ypts2 = upper_bend2_y + lower_bend2_y[::-1]
    bend_ypts3 = upper_bend3_y + lower_bend3_y[::-1]
    bend_ypts4 = upper_bend4_y + lower_bend4_y[::-1]
    
    D = Device('Rapid_Adiabatic_Coupler')
    D.add_polygon(points = (xpts,ypts), layer = 0)  
    D.add_polygon(points = (xpts2,ypts2), layer = 0)     
    D.add_polygon(points = (bend_xpts1, bend_ypts1), layer = 0)
    D.add_polygon(points = (bend_xpts1, bend_ypts2), layer = 0)
    D.add_polygon(points = (bend_xpts2, bend_ypts3), layer = 0)
    D.add_polygon(points = (bend_xpts2, bend_ypts4), layer = 0)
    
    D.add_port(name = 'wgport1', midpoint = [bend_x1[0], width1_ans[0]/2 + gap/2 + bend1[0] + np.tan(theta[0])*length/len(xaxis)], width = width1_ans[0], orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [bend_x1[0], -width2_ans[0]/2 - gap/2 + bend2[0] + np.tan(theta[0])*length/len(xaxis)], width = width2_ans[0], orientation = 180)
    D.add_port(name = 'wgport3', midpoint = [bend_x2[-1], width1_ans[-1]/2 + gap/2 + bend3[-1] + np.tan(theta[-1])*length/len(xaxis)], width = width1_ans[-1], orientation = 0)
    D.add_port(name = 'wgport4', midpoint = [bend_x2[-1], -width2_ans[-1]/2 - gap/2 + bend4[-1] + np.tan(theta[-1])*length/len(xaxis)], width = width2_ans[-1], orientation = 0)
    D.info['length'] = xaxis[len(xaxis)-1] + bend_length1 + bend_length2
    
#  creat port 
    D1 = Device('Add ports')
    R_A_DC = D1.add_ref(D)
    # L_input1 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    # L_input1.y = port_gap12/2 
    # L_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    # L_input2.y=-port_gap12/2
    
    # R_input1 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    # R_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    

    # R_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    
    
    # R_A_DC.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    # R_A_DC.x = R_A_DC.x + buffer_length
    # R_A_DC.y = R_A_DC.y -port_gap12/2
    # R_input1.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    # R_input1.x=R_input1.x+R_A_DC.xsize+buffer_length*2
    # R_input1.y = port_gap34/2
    # R_input2.x=R_input1.x
    # R_input2.y=-port_gap34/2
    
    
    L_input1b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input1b.connect(port='wgport2', destination=R_A_DC.ports['wgport1'])
    L_input1b.move([-30,0])
    L_input2b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input2b.connect(port='wgport2', destination=R_A_DC.ports['wgport2'])
    L_input2b.move([-30,0])
    # L_input2.y=-port_gap12/2

    R_input1b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    R_input1b.connect(port='wgport1', destination=R_A_DC.ports['wgport3'])
    R_input1b.move([1,0])
    
    
    R_input2b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    R_input2b.connect(port='wgport1', destination=R_A_DC.ports['wgport4'])
    R_input2b.move([1,0])
    
    # D1.add_ref(pr.route_basic(port1 = L_input1.ports['wgport2'], port2 = L_input1b.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    # D1.add_ref(pr.route_basic(port1 = L_input2.ports['wgport2'], port2 = L_input2b.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    # D1.add_ref(pr.route_basic(port1 = R_input1.ports['wgport1'], port2 = R_input1b.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    # D1.add_ref(pr.route_basic(port1 = R_input2.ports['wgport1'], port2 = R_input2b.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    
    
    
    D1.add_ref(pr.route_basic(port1 = L_input1b.ports['wgport2'], port2 = R_A_DC.ports['wgport1'], path_type = 'straight', width_type = 'straight'))
    D1.add_ref(pr.route_basic(port1 = L_input2b.ports['wgport2'], port2 = R_A_DC.ports['wgport2'], path_type = 'straight', width_type = 'straight'))
    D1.add_ref(pr.route_basic(port1 = R_input1b.ports['wgport1'], port2 = R_A_DC.ports['wgport3'], path_type = 'straight', width_type = 'straight'))
    D1.add_ref(pr.route_basic(port1 = R_input2b.ports['wgport1'], port2 = R_A_DC.ports['wgport4'], path_type = 'straight', width_type = 'straight'))
    
    
    D1.add_port(name = 'wgport1', midpoint = [L_input1b.xmin,L_input1b.y], width = width_ref, orientation = 180)
    D1.add_port(name = 'wgport2', midpoint = [L_input2b.xmin,L_input2b.y], width = width_ref, orientation =180)
    D1.add_port(name = 'wgport3', midpoint = [R_input1b.xmax,R_input1b.y], width = width_ref, orientation =0)
    D1.add_port(name = 'wgport4', midpoint = [R_input2b.xmax,R_input2b.y], width = width_ref, orientation =0)
    
    return D1



def Rapid_Adiabatic_DC_sMZI(filepath ='C:/Users/kist/Documents/KIST 허형준/Fabrication/E-beam design/2023 EBL design/230421_adiabatic_DC/LN_design_code/Adia_DC.csv', length = 100, gap = 0.6, port_gap12 = 400, port_gap34 = 100, width_ref = 0.6, height = 1.5, bend_length1 = 200, bend_length2 = 30, buffer_length = 1000, ref_Length=1, N_MZI = 1):
    """ Creates an Rapid_Adiabatic_Coupler"""
    
    D = Device('Rapid_Adiabatic_DC_sMZI')
    xx = 0 
    yy = 0
    for i in range (N_MZI):    
        R_A_DCL = D.add_ref(Rapid_Adiabatic_DC_v003(filepath = filepath, length = length, gap = gap, port_gap12 =  port_gap12, port_gap34 =  port_gap34, width_ref = width_ref, height = height, bend_length1 = bend_length1, bend_length2 = bend_length2, buffer_length = buffer_length , ref_Length=ref_Length))
        R_A_DCL.xmin = xx
        R_A_DCL.ymin = yy
        if i ==0:
            D.add_port(name = 'wgport1', midpoint = [R_A_DCL.ports['wgport1'].x,R_A_DCL.ports['wgport1'].y], width = width_ref, orientation =180)
            D.add_port(name = 'wgport2', midpoint = [R_A_DCL.ports['wgport2'].x,R_A_DCL.ports['wgport2'].y], width = width_ref, orientation =180)
            
            
        R_A_DCR = D.add_ref(Rapid_Adiabatic_DC_v003(filepath = filepath, length = length, gap = gap, port_gap12 =  port_gap12, port_gap34 =  port_gap34, width_ref = width_ref, height = height, bend_length1 = bend_length1, bend_length2 = bend_length2, buffer_length = buffer_length , ref_Length=ref_Length))
        R_A_DCR.mirror([0,1])
        R_A_DCR.connect(port = 'wgport3', destination=R_A_DCL.ports['wgport3'])
        
        xx = R_A_DCR.xmax
        yy = R_A_DCR.ymin
    
    D.add_port(name = 'wgport3', midpoint = [R_A_DCR.ports['wgport1'].x,R_A_DCR.ports['wgport1'].y], width = width_ref, orientation =0)
    D.add_port(name = 'wgport4', midpoint = [R_A_DCR.ports['wgport2'].x,R_A_DCR.ports['wgport2'].y], width = width_ref, orientation =0)
    
    
    
    return D


def Rapid_Adiabatic_DC_sMZI_v002(filepath ='C:/Users/kist/Documents/KIST 허형준/Fabrication/E-beam design/2023 EBL design/230421_adiabatic_DC/LN_design_code/Adia_DC.csv', length = 100, gap = 0.6, port_gap12 = 400, port_gap34 = 100, width_ref = 0.6, height = 1.5, bend_length1 = 200, bend_length2 = 30, buffer_length = 1000, ref_Length=1, N_MZI = 1):
    """ Creates an Rapid_Adiabatic_Coupler"""
    
    D = Device('Rapid_Adiabatic_DC_sMZI')
    xx = 0 
    yy = 0
    for i in range (N_MZI):    
        R_A_DCL = D.add_ref(Rapid_Adiabatic_DC_v003(filepath = filepath, length = length, gap = gap, port_gap12 =  port_gap12, port_gap34 =  port_gap34, width_ref = width_ref, height = height, bend_length1 = bend_length1, bend_length2 = bend_length2, buffer_length = buffer_length , ref_Length=ref_Length))
        R_A_DCL.xmin = xx
        R_A_DCL.ymin = yy
        if i ==0:
            D.add_port(name = 'wgport1', midpoint = [R_A_DCL.ports['wgport1'].x,R_A_DCL.ports['wgport1'].y], width = width_ref, orientation =180)
            D.add_port(name = 'wgport2', midpoint = [R_A_DCL.ports['wgport2'].x,R_A_DCL.ports['wgport2'].y], width = width_ref, orientation =180)
            
            
        R_A_DCR = D.add_ref(Rapid_Adiabatic_DC_v003(filepath = filepath, length = length, gap = gap, port_gap12 =  port_gap12, port_gap34 =  port_gap34, width_ref = width_ref, height = height, bend_length1 = bend_length1, bend_length2 = bend_length2, buffer_length = buffer_length , ref_Length=ref_Length))
        R_A_DCR.mirror([0,1])
        R_A_DCR.connect(port = 'wgport3', destination=R_A_DCL.ports['wgport3'])
        
        xx = R_A_DCR.xmax
        yy = R_A_DCR.ymin
    
    D.add_port(name = 'wgport3', midpoint = [R_A_DCR.ports['wgport1'].x,R_A_DCR.ports['wgport1'].y], width = width_ref, orientation =0)
    D.add_port(name = 'wgport4', midpoint = [R_A_DCR.ports['wgport2'].x,R_A_DCR.ports['wgport2'].y], width = width_ref, orientation =0)
    
    
   
#  creat port 
    D1 = Device('Add ports')
    R_A_DC = D1.add_ref(D)
    L_input1 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input1.y = port_gap12/2 
    L_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input2.y=-port_gap12/2
    
    R_input1 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    R_input2 = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    
    
    R_A_DC.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    R_A_DC.x = R_A_DC.x + buffer_length
    R_A_DC.y = R_A_DC.y -port_gap12/2
    R_input1.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    R_input1.x=R_input1.x+R_A_DC.xsize+buffer_length*2
    R_input1.y = port_gap34/2
    R_input2.x=R_input1.x
    R_input2.y=-port_gap34/2
    
    
    L_input1b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input1b.connect(port='wgport2', destination=R_A_DC.ports['wgport1'])
    L_input1b.move([-30,0])
    L_input2b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    L_input2b.connect(port='wgport2', destination=R_A_DC.ports['wgport2'])
    L_input2b.move([-30,0])
    # L_input2.y=-port_gap12/2

    R_input1b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    R_input1b.connect(port='wgport1', destination=R_A_DC.ports['wgport3'])
    R_input1b.move([30,0])
    
    
    R_input2b = D1.add_ref(waveguide(length = ref_Length, width=width_ref))
    R_input2b.connect(port='wgport1', destination=R_A_DC.ports['wgport4'])
    R_input2b.move([30,0])
    
    D1.add_ref(pr.route_basic(port1 = L_input1.ports['wgport2'], port2 = L_input1b.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = L_input2.ports['wgport2'], port2 = L_input2b.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input1.ports['wgport1'], port2 = R_input1b.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input2.ports['wgport1'], port2 = R_input2b.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    
    
    
    D1.add_ref(pr.route_basic(port1 = L_input1b.ports['wgport2'], port2 = R_A_DC.ports['wgport1'], path_type = 'straight', width_type = 'straight'))
    D1.add_ref(pr.route_basic(port1 = L_input2b.ports['wgport2'], port2 = R_A_DC.ports['wgport2'], path_type = 'straight', width_type = 'straight'))
    D1.add_ref(pr.route_basic(port1 = R_input1b.ports['wgport1'], port2 = R_A_DC.ports['wgport3'], path_type = 'straight', width_type = 'straight'))
    D1.add_ref(pr.route_basic(port1 = R_input2b.ports['wgport1'], port2 = R_A_DC.ports['wgport4'], path_type = 'straight', width_type = 'straight'))
    
    
    D1.add_port(name = 'wgport1', midpoint = [L_input1.xmin,port_gap12/2], width = width_ref, orientation = 180)
    D1.add_port(name = 'wgport2', midpoint = [L_input1.xmin,-port_gap12/2], width = width_ref, orientation =180)
    D1.add_port(name = 'wgport3', midpoint = [R_input1.xmax, port_gap34/2], width = width_ref, orientation =0)
    D1.add_port(name = 'wgport4', midpoint = [R_input1.xmax,-port_gap34/2], width = width_ref, orientation =0)
    
    return D1


def Linear_Adiabatic_DC(width1=0.9, width2=0.6, length1=2, length2=2,gap_s=0.65,gap_e=0.5, port_gap12 = 400, port_gap34 = 400, buffer_length = 1000, a=0, b=0,ref_Length=1, layer = 0):
    """ Creates an Adiabatic_DC"""

    t1 = np.linspace(0, length1, int(length1)*10)
    t2 = np.linspace(0, length2, int(length2)*10)
    t = np.linspace(0, length1+length2, int(length1+length2)*10)
    xaxis=(t).tolist()
    width_ans=(width1-0*t1).tolist()+(width1-(width1-width2)/length2*t2).tolist()
    gap_ans= (gap_s-(gap_s-gap_e)/(length1)*t1).tolist()+(gap_e+0*t2).tolist()

    upper_points_x = (t).tolist()
    upper_points_y = (np.interp(t, xaxis,width_ans)+np.interp(t, xaxis,gap_ans)/2).tolist()
    lower_points_x = (t).tolist()
    lower_points_y = (np.interp(t, xaxis,gap_ans)/2).tolist()

    upper_points_x2 = (t).tolist()
    upper_points_y2 = (-np.interp(t, xaxis,gap_ans)/2).tolist()
    lower_points_x2 = (t).tolist()
    lower_points_y2 = (-np.interp(t, xaxis,gap_ans)/2-width2).tolist()
    xpts = upper_points_x + lower_points_x[::-1]
    ypts = upper_points_y + lower_points_y[::-1]
    xpts2 = upper_points_x2 + lower_points_x2[::-1]
    ypts2 = upper_points_y2 + lower_points_y2[::-1]

    D = Device('Adiabatic_DC')
    D.add_polygon(points = (xpts,ypts), layer = 0)  
    D.add_polygon(points = (xpts2,ypts2), layer = 0)     
    D.add_port(name = 'wgport1', midpoint = [xaxis[0],width_ans[0]/2+gap_ans[0]/2], width = width_ans[0], orientation = 180)
    D.add_port(name = 'wgport2', midpoint = [xaxis[0],-width2/2-gap_ans[0]/2], width = width2, orientation = 180)
    D.add_port(name = 'wgport3', midpoint = [xaxis[len(xaxis)-1],width_ans[len(width_ans)-1]/2+gap_ans[len(gap_ans)-1]/2], width = width_ans[len(width_ans)-1], orientation = 0)
    D.add_port(name = 'wgport4', midpoint = [xaxis[len(xaxis)-1],-width2/2-gap_ans[len(gap_ans)-1]/2], width = width2, orientation = 0)
    D.info['length'] = xaxis[len(xaxis)-1]
#  creat port 
    D1 = Device('Add ports')
    L_A_DC = D1.add_ref(D)
    L_input1 = D1.add_ref(waveguide(length = ref_Length, width=width2))
    L_input1.y=port_gap12/2 
    L_input2 = D1.add_ref(waveguide(length = ref_Length, width=width2))
    L_input2.y=-port_gap12/2

    R_input1 = D1.add_ref(waveguide(length = ref_Length, width=width2))
    R_input2 = D1.add_ref(waveguide(length = ref_Length, width=width2))
    L_A_DC.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    L_A_DC.x = L_A_DC.x+buffer_length
    L_A_DC.y = L_A_DC.y -port_gap12/2+a
    R_input1.connect(port="wgport1",destination = L_input1.ports['wgport2'] )
    R_input1.x=R_input1.x+L_A_DC.xsize+buffer_length*2
    R_input2.y=R_input1.y-port_gap34-b 
    R_input2.x=R_input1.x
    D1.add_ref(pr.route_basic(port1 = L_input1.ports['wgport2'], port2 = L_A_DC.ports['wgport1'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = L_input2.ports['wgport2'], port2 = L_A_DC.ports['wgport2'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input1.ports['wgport1'], port2 = L_A_DC.ports['wgport3'], path_type = 'sine', width_type = 'sine'))
    D1.add_ref(pr.route_basic(port1 = R_input2.ports['wgport1'], port2 = L_A_DC.ports['wgport4'], path_type = 'sine', width_type = 'sine'))
    
    D1.add_port(name = 'wgport1', midpoint = [L_input1.xmin,port_gap12/2], width = width2, orientation = 180)
    D1.add_port(name = 'wgport2', midpoint = [L_input1.xmin,-port_gap12/2], width = width2, orientation =180)
    D1.add_port(name = 'wgport3', midpoint = [R_input1.xmax,b + port_gap34/2], width = width2, orientation =0)
    D1.add_port(name = 'wgport4', midpoint = [R_input1.xmax,b - port_gap34/2], width = width2, orientation =0)
    
    return D1
# Wwg=0.6 # waveguide width
# Twg=2.5 # taper waveguide width
# T_length=100
# MMI_length22=71.2
# MMI_width22 = 5
# MMI_length12= 11
# MMI_width12 = 3.75
# MZI_gap=200
# device_gap=450
# buffer_length=1000
# buffer_length1=400
# port_gap=400
# wt_gap=20
# device_length= 14000
# print("device_length",device_length)
# r_bending=200
# electrode_width=10
# electrode_gap=5
# E_pad=100
# EO_length=6000
# test=Device("test")
# a=test.add_ref(PSR_adiabatic(WGwidth = 1, WGport1s= 2.26,WGport2s= 1.57, WGport1e= 3.5,WGport2e= 0.95,WGT1=2,WGT2=1.6,conv_length=500,coupling_length=1600, coupling_gap = 0.6, port_gap12 = 301.1, Taper_buffer_length = 100,Taper_buffer_length2=100, buffer_length = 1000, a=0,ref_Length=1, layer = 0))
# qp(test)

# test.write_gds("test.gds")
