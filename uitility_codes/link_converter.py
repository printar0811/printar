# Read the links from the file
with open("links.txt", "r") as file:
    links = file.readlines()

# Strip newline characters from each link and create the array
links_array = [link.strip() for link in links]

# Write the array to a new file
with open("output.txt", "w") as file:
    file.write("array=" + str(links_array))

print("Array created and saved to output.txt")

