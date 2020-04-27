Readme:

Steps to run the project:

Download the Project folder MovieRecommendationSystem
Go to terminal and change the directory to MovieRecommendationSystem
The folder has the jar file as well to run the project.
 From the terminal execute  the command spark-submit --class kiran.MovieSimilarities movierecommendationsystem_2.11-0.1.jar 50
We get the top 10 movies similar to Star Wars (1977)

Notes:
Class name is MovieSimilarities which is inside package ‘kiran’.
movierecommendationsystem_2.11-0.1.jar is the jar file name.
50 is the movieId of  Star Wars (1977), we are looking for similar movies to Star Wars (1977).