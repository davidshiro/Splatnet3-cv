import csv

# string similarity algorithm (edit distance, dynamic programming)
def edit_distance(str1, str2):  
    m, n = len(str1), len(str2)  
    dp = [[0] * (n + 1) for _ in range(m + 1)]  
    for i in range(m + 1):  
        for j in range(n + 1):  
            if i == 0:  
                dp[i][j] = j  
            elif j == 0:  
                dp[i][j] = i  
            elif str1[i - 1] == str2[j - 1]:  
                dp[i][j] = dp[i - 1][j - 1]  
            else:  
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])  

    return dp[m][n]  

class TeamDetector:
    def __init__(self, filename):
        self.teamdict = {}

        #populate dictionary with roster names
        with open(filename) as csvfile:
            rreader = csv.DictReader(csvfile)
            for row in rreader:
                #print(row['Team Name'])
                self.teamdict[row['Team Name']] = [row['Player 1'], row['Player 2'], row['Player 3'], row['Player 4']]
                # add players beyond 5 if present, helps to optimize search
                for i in range(5,9):
                    tar_col = "Player %d" %(i)
                    if (row[tar_col] != ''):
                        #print(tar_col)
                        #print(row[tar_col])
                        self.teamdict[row['Team Name']].append(row[tar_col])
                    else:
                        break
                #print(self.teamdict[row['Team Name']])

    #searches all possible teams for name
    def name_search(self, query):
        #format: Team Name, player number, edit distance, actual name
        found = ["team name", 0, 100, "name"]
        #loop through every registered name
        for team_name in self.teamdict:
            #allows loop to access index while iterating
            for i, name in enumerate(self.teamdict[team_name]):
                #removes splashtag identifiers (last 5 char)
                detagged_name = name[:-5]
                dis = edit_distance(query, detagged_name)
                if dis < found[2]:
                    found = [team_name, i, dis, name]
                elif dis == found[2]:
                    #print("Found potential other name:")
                    #print(name, dis)
                    pass
        #print(found)
        return(found)
    
    #searches for name given known team
    def name_search_team(self, query, team_name):
        found = [team_name, 0, 100, "name"]
        for i, name in enumerate(self.teamdict[team_name]):
            #removes splashtag identifiers (last 5 char)
            detagged_name = name[:-5]
            dis = edit_distance(query, detagged_name)
            if dis < found[2]:
                found = [team_name, i, dis, name]
            elif dis == found[2]:
                #print("Found potential other name:")
                #print(name, dis)
                pass
        #print(found)
        return(found)




#print("name to search:")
#input_query = input()
#td = TeamDetector('rosters.csv')
#td.name_search(input_query)