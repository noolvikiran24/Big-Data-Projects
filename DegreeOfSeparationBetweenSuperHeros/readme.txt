Readme:

1. Extract the zip file.
2. Open intellij and open DegreeOfSeparationBetweenSuperHeros
3. Generate the jar file.
4. Open terminal go to the location of the BigDataFinalProject
5. Run the command spark-submit --class bigdataproject.PopularSuperHeroCalculation 'jar file location' 'location of MarvelGraphData.txt' 'MarvelNamesData.txt' 'ResultPath' startingSuperHeroId targetSuperHeroId
5. If the startingSuperHeroId targetSuperHeroId are not given, then by default the program will run with startingSuperHeroId=5306(SPIDER-MAN/PETER PAR) targetSuperHeroId=6306(WOLVERINE/LOGAN)
6. The Result with the top 5 famous heros using graphX, without using graphX and degree of separation between startingSuperHero to targetSuperHero will be shown.
7. We can execute with similar cluster steps setup on AWS. We have executed the code in AWS as well.

Note:
1. The Super hero id with names are available in MarvelNamesData.txt' and MarvelGraphData.txt has the connections of each super hero.
2. Demo results can be seen in part-0003 in Results folder.
3. Example command to run: spark-submit --class bigdataproject.PopularSuperHeroCalculation big-data-final-project_2.11-0.1.jar MarvelGraphData.txt MarvelNamesData.txt Results 5306 5659


Path for MarvelGraphData s3://bigdatafinalprojectdata/MarvelGraphData.txt
Path for MarvelNamesData s3://bigdatafinalprojectdata/MarvelNamesData.txt

