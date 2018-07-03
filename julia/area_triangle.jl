function area_triangle(a,b,c)
a,b,c=reverse(sort([a,b,c]))
area = .25*sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))
return area
end
