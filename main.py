from my_modules.csv_writer import csv_writer

writer = csv_writer("teste.csv", ["param1", "param2"])
for i in range(20):
	writer.write_line(['"string"', "12345"])
writer.close()
