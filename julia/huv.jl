# Experiment with different ways to compute Huv:


function Huv_alt(l_max,lam)
Huv = zeros(typeof(lam),l_max+3,l_max+1)
clam = cos(lam); slam = sin(lam)
clam2 = clam*clam; clamn = clam; slamn = slam
for u=0:2:l_max+2
  if u == 0
    Huv[1,1]=  2*lam+pi
    Huv[1,2]= -2*clam
    slamn = slam
    v=2
    while v <= l_max
      Huv[1,v+1]= (-2*clam*slamn+(v-1)*Huv[1,v-1])/(u+v)
      println("lam: ",lam," u: ",u," v: ",v," f1: ",-2*clam*slamn/(u+v)," f2: ",(v-1)/(u+v)*Huv[1,v-1]," Huv: ",Huv[1,v+1])
      slamn *= slam
      v+=1
    end
  else
    slamn = slam
    v=0
    Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
    println("lam: ",lam," u: ",u," v: ",v," f1: ",2*clamn*slamn/(u+v)," f2: ",(u-1)/(u+v)*Huv[u-1,v+1]," Huv: ",Huv[u+1,v+1])
    slamn *= slam
    v=1
    slamn *= slam
    v=1
    Huv[u+1,v+1] = -2*clamn*clam2/(u+1)
#    Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
    println("lam: ",lam," u: ",u," v: ",v," f1: ",2*clamn*slamn/(u+v)," f2: ",(u-1)/(u+v)*Huv[u-1,v+1]," Huv: ",Huv[u+1,v+1])
    slamn = slam
    clamn *= clam2
    v=2
    while v <= l_max
#      Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
      Huv[u+1,v+1]= (-2*clamn*slamn+(v-1)*Huv[u+1,v-1])/(u+v)
      println("lam: ",lam," u: ",u," v: ",v," f1: ",-2*clamn*slamn/(u+v)," f2: ",(v-1)/(u+v)*Huv[u+1,v-1]," Huv: ",Huv[u+1,v+1])
      slamn *= slam
      v+=1
    end
#    clamn *= clam2
  end
end
return Huv
end

function Huv_compute(l_max,lam)
Huv = zeros(typeof(lam),l_max+3,l_max+1)
clam = cos(lam); slam = sin(lam)
clam2 = clam*clam; clamn = clam; slamn = slam
for u=0:2:l_max+2
  if u == 0
    Huv[1,1]=  2*lam+pi
    Huv[1,2]= -2*clam
    slamn = slam
    v=2
    while v <= l_max
      Huv[1,v+1]= (-2*clam*slamn+(v-1)*Huv[1,v-1])/(u+v)
      println("lam: ",lam," u: ",u," v: ",v," f1: ",-2*clam*slamn/(u+v)," f2: ",(v-1)/(u+v)*Huv[1,v-1]," Huv: ",Huv[1,v+1])
      slamn *= slam
      v+=1
    end
  else
    slamn = slam
    v = 0
    while v <= l_max
      Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
      println("lam: ",lam," u: ",u," v: ",v," f1: ",2*clamn*slamn/(u+v)," f2: ",(u-1)/(u+v)*Huv[u-1,v+1]," Huv: ",Huv[u+1,v+1])
      slamn *= slam
      v+=1
    end
    clamn *= clam2
  end
end
return Huv
end

l_max = 10
epsilon = 1e-3
lam = -pi/2+epsilon
Huv = Huv_compute(l_max,lam)
Huv2= Huv_alt(l_max,lam)
println("Huv diff: ",Huv-Huv2)
lam = pi/2-epsilon
Huv = Huv_compute(l_max,lam)
Huv2= Huv_alt(l_max,lam)
println("Huv diff: ",Huv-Huv2)
