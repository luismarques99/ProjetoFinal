class csv_writer:
    """Class that represents the structure of a CSV file writer"""

    def __init__(self, filename="", header=list()):
        """Creates an instance of a CSV file writer

        Args:
            filename (str, required): string to define the name of the file. Defaults to "" (empty string).
            header (str[], required): list of strings to be written as the header of the file. Defaults to list() (empty list).
        """
        self.file = open(filename, "w")
        self.write_line(header)

    def write_at_once(self, content: list):
        """Writes a list of string's list at once in the file

        :param content (list): list of string's list
        """
        for element in content:
            self.write_line(element)

    def write_line(self, content=list()):
        """Writes a list of strings as a line in the file

        Args:
            content (str[], required): list of strings to be written into the file. Defaults to list() (empty list).
        """
        self.file.write(content[0])
        for index in range(len(content) - 1):
            self.file.write(",")
            self.file.write(content[index + 1])
        self.file.write("\n")

    def close(self):
        """Closes the file"""
        self.file.close()
