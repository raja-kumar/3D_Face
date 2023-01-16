from getopt import getopt
import os
import sys

def select_2_imgs(person):
    names = []
    if person == '1':
        names = ['1', '10']
    elif person == '2':
        names = ['17', '21']
    elif person == '3':
        names = ['17', '21']
    elif person == '4':
        names = ['1', '10']
    elif person == '5':
        names = ['16', '20']
    elif person == '6':
        names = ['17', '20']
    elif person == '7':
        names = ['17', '21']
    elif person == '9':
        names = ['17', '21']
    elif person == '13':
        names = ['51', '46']
    if person == '16':
        names = ['1', '10']
    if person == '17':
        names = ['17', '21']
    if person == '18':
        names = ['1', '10']
    if person == '19':
        names = ['1', '10']
    elif person == '122':
        names = ['40', '44']
    elif person == '212':
        names = ['47', '51']
    return names

def select_3_imgs(person):

    names = []
	if person == '1':
		names = ['1', '10', '8']
	elif person == '2':
		names = ['17', '21', '10']
	elif person == '3':
		names = ['17', '21', '10']
	elif person == '4':
		names = ['1', '10', '8']
	elif person == '5':
		names = ['16', '20', '9']
	elif person == '6':
		names = ['17', '20', '10']
	elif person == '7':
		names = ['17', '21', '7']
	elif person == '9':
		names = ['17', '21', '10']
	elif person == '13':
		names = ['51', '46', '47']
	elif person == '16':
		names = ['1', '10', '8']
	elif person == '17':
		names = ['17', '21', '10']
	elif person == '18':
		names = ['1', '10', '8']
	elif person == '19':
		names = ['1', '10', '8']
	elif person == '122':
		names = ['40', '44', '45']
	elif person == '212':
		names = ['47', '51', '52']
	return names



