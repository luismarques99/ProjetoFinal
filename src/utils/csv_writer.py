class CSVWriter:
    """Class that represents the structure of a CSV file writer"""

    def __init__(self, filename: str, header: tuple, file_type: str = "w"):
        """Creates an instance of a CSV file writer

        Args:
            filename (str): string to define the name of the file.
            header (tuple): tuple of strings to be written as the header of the file.
        """
        self.file = open(filename, file_type)
        self.write_line(header)

    def write_at_once(self, content: list):
        """Writes a list of string's list at once in the file

        Args:
            content (list): list of string's list
        """
        for element in content:
            self.write_line(element)

    def write_line(self, content: tuple):
        """Writes a tuple of strings as a line in the file

        Args:
            content (tuple): tuple of strings to be written into the file.
        """
        self.file.write(content[0])
        for index in range(len(content) - 1):
            self.file.write(",")
            self.file.write(content[index + 1])
        self.file.write("\n")

    def close(self):
        """Closes the file"""
        self.file.close()
