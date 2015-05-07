function newnorm = process_normap(Nmap)

    Nmap = im2single(Nmap);
    newnorm(:,:,1) = (2/255)* Nmap(:,:,3)-1;
    newnorm(:,:,2) = (2/255)* Nmap(:,:,2)-1;
    newnorm(:,:,3) = (2/255)* Nmap(:,:,1)-1;

end