### new version
``````
n0(0,60)          n1(1000,60)       n2(2000,60)       n3(3000,60)       n4(4000,60)       n5(5000,60)      
⦿----------------⦿----------------⦿----------------⦿----------------⦿----------------⦿
                 /                   \               /                  \                
                /                     \             /                    \              
               ⦿                      ⦿          ⦿                      ⦿          
          n8(700,0)               n9(2300, 0)    n10(2700,0)          n11(4300, 0)    
                    
`````` 





### Old version
````` 
n0(0,60)          n1(1000,60)       n2(2000,60)       n3(3000,60)       n4(4000,60)       n5(5000,60)      n6(6000,60)       n7(7000,60)
⦿----------------⦿----------------⦿----------------⦿----------------⦿----------------⦿----------------⦿----------------⦿ 
                 /                   \               /                  \                /                  \
                /                     \             /                    \              /                    \
               ⦿                      ⦿          ⦿                      ⦿          ⦿                      ⦿ 
          n8(700,0)               n9(2300, 0)    n10(2700,0)          n11(4300, 0)  n12(4700,0)           n13(6300, 0)
                    
`````` 

how simultaion has been built :
1. create Node file
2. Create edge and ege_type files 
3. generate network using this command:

netconvert --node-files nodes.nod.xml --edge-files edge.edg.xml -t edge_type.type.xml -o network.net.xml


netconvert --node-files nodes.nod.xml --edge-files edge.edg.xml -t edge_type.type.xml -o network.net.xml   --default.junctions.radius 0 
4. generate demande using this command 
python sumo_tools/randomTrips.py -n data/network.net.xml -e 50
5. for visualition create setting file 


  
