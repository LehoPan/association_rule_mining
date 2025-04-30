# association_rule_mining
Dependencies: ```mlxtend``` ```pandas``` ```numpy``` ```networkx```

## Running
Run RuleMining.py for the dataset online retail II. It will show you a graphical representation of the rules mined, and after closing it will prompt you for stock ID's to be input and it will recommend you highly correlated items to bundle together based on the rules extracted. 

The initial running on the data can take quite a while, for my desktop it took 5-8 minutes every time. ```testdata.xlsx``` is just the first 29 rows of the data set to speed up during testing. You can switch the input on lines 9-11 of the code to the test data if you just want to quickly poke around.