while (True):
	try:
		x = int(input())
	except ValueError:
		print("Please input an integer")
	else:
		break

print(x)
