todoList = ["掃地", "拖地", "煮飯"]
notyet = []
for idx, task in enumerate(todoList):
    answer = input("您的第{}件事是{},您完成了嗎?Ans:".format(idx+1, task))
    if answer != "done":
        notyet.append(task)
print("您還剩下以下工作:")
print(notyet)