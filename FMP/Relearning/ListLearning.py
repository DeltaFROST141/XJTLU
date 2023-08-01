shopping_list = ["Giant/Liv", "Keyboard", "Mouse", "iPad Pro"]

shopping_list.append("Airpods Max")
shopping_list.remove("Keyboard")

#

for items in shopping_list:
    print(f"I would like to buy {items}")

print(shopping_list)
print(len(shopping_list))

GSW_num_list = [9, 11, 23, 30, 35]
print(max(GSW_num_list))
print(min(GSW_num_list))
print(sorted(GSW_num_list))