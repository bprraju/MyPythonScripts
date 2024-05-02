import argparse
from netaddr import *
import os
import sys

def main():
	parser = argparse.ArgumentParser(description="Subtract subnets from supernet\n") 
	parser.add_argument('--supernet', required=True,type=str, help=' Enter supernet ')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--file', type=str, help=' Enter file which contains subnets on each line')
	group.add_argument('--subnets',type=str, help=' Enter list of subnets ')
	args = parser.parse_args() 

	lines = []
	supernets = args.supernet

	if args.file is None:					
		subnets_list = args.subnets.split(",")
	if args.subnets is None:
		subnets_file = args.file
		if os.path.exists(subnets_file) is False:
			print(subnets_file, " doesn't exist")
			sys.exit()
		if os.path.exists(subnets_file):
			with open(subnets_file, 'rb') as f:
				try:
					lines = [line.strip() for line in f]
				except IOError:
					print("Could not read file:", subnets_file)					
			subnets_list = lines
			if len(subnets_list) == 0:
				print(subnets_file, " content is empty")
				sys.exit()

	s1 = IPSet()
	s1.add(supernets)
#	sub1.add(supernets)

	print("\nYou have Entered supernet", supernets)
	print("\nSubnets to be removed are:")
	for i in subnets_list:
		print(i)
		s1.remove(i)
	

	print("\n Result is: ", s1)
	print("\n")





if __name__ == "__main__":
    main()
