### new version
``````
n0(0,60)          n1(1000,60)       n2(2000,60)       n3(3000,60)       n4(4000,60)       n5(5000,60)      
⦿----------------⦿----------------⦿----------------⦿----------------⦿----------------⦿
                 /                   \               /                  \                
                /                     \             /                    \              
               ⦿                      ⦿          ⦿                      ⦿          
          n8(700,0)               n9(2300, 0)    n10(2700,0)          n11(4300, 0)    
                    
`````` 



how simultaion has been built :
1. create Node file
2. Create edge and ege_type files 
3. generate network using this command:

netconvert --node-files nodes.nod.xml --edge-files edge.edg.xml -t edge_type.type.xml -o network.net.xml



  
