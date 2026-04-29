a = 10
b = 3.14
c = "hello"# 字符串 str
d = True # 布尔 bool

print(a)
print(b)
print(c)
print(d)

print(type(a))
print(type(c))

e = a+5
print("a+5=",e)

name = "小明"
greeting = "你好，" + name
print(greeting)

f = float(a)
print(f)
print (type(f))
#练习
x = 5
y = 3
print("x+y=",x+y)
print("x-y=",x-y)
print("x*y=",x*y)
print("x/y=",x/y)
pi = 3.14159
r = 5
print("圆的面积为",pi*r**2)
name2 = "张三"
age = 25
print(name2,"今年",age,"岁")
print(name2+"今年"+str(age)+"岁")

ages = [23,30,45,28,35]
print("所有人的年龄：",ages)

print("第1个人:",ages[0])
print("第2个人:",ages[1])
print("最后一个人:",ages[-1])# 35（-1表示倒数第1）
# 切片：取一部分 [start:end]，end不包括
print("前3个人:",ages[0:3])
print("从第2个到最后:",ages[1:])
# 添加元素
ages.append(40)
print("添加后:",ages)
# 修改元素
ages[0]=100
print("修改第1个:", ages)
# 列表长度
print("总人数:", len(ages))

scores=[85,92,78,90,88]
print("第三个成绩:",ages[2])
print("平均分是:",sum(scores)/len(scores))
scores[-1]=100
print("修改后成绩:",scores)
# ========== 字典 dict ==========
# 字典 = 带名字的盒子，通过"键"找"值"
person = {"name":"张三",
          "age":30,
          "city":"北京"
}
print("姓名:",person["name"])
print("年龄:",person["age"])
# 添加新键值对
person["job"]="工程师"
print("添加后:",person)
person["age"]=31
print("修改后:",person)
# 遍历字典----#person.items()把字典变成一串"键值对"，像 [("name","张三"), ("age",30), ...]
for key,value in person.items():#for ... in ...	循环：逐个取出里面的东西
    print(key,":",value)#key, value	解包：每个东西是一对 (键, 值)，自动拆开，左边给 key，右边给 value
#第一次循环：key="name", value="张三";第二次循环：key="age", value=30;第三次循环：key="city", value="北京"
student = {"name":"光头强",
           "score":"59",
           "subject":"伐木课"
}
print("姓名:",student["name"])
print("成绩:",student["score"])
student["score"]=69
for key,value in student.items():
    print(key,":",value)












