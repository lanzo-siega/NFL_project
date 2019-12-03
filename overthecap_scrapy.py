# -*- coding: utf-8 -*-
import scrapy

class OverthecapSpider(scrapy.Spider):
    name = 'overthecap'
    allowed_domains = ['overthecap.com']
    start_urls = ['http://overthecap.com/contracts']

    def parse(self, response):
        for table in response.xpath('/html/body/div[1]/div/div/div[3]/table/tbody'):
            Player = table.css('tr:nth-child(n) td:nth-child(1) a::text').getall()
            Position = table.css('tr:nth-child(n) td:nth-child(2)::text').getall()
            Team = table.css('tr:nth-child(n) td:nth-child(3) a::text').getall()
            TotalValue = table.css('tr:nth-child(n) td:nth-child(4)::text').getall()
            AvgPerYear = table.css('tr:nth-child(n) td:nth-child(5)::text').getall()
            TotGuarantee = table.css('tr:nth-child(n) td:nth-child(6)::text').getall()
            AvgGuarPerYear = table.css('tr:nth-child(n) td:nth-child(7)::text').getall()
            PercentGuaranteed = table.css('tr:nth-child(n) td:nth-child(8)::text').getall()
            
        for item in zip(Player,Position,Team,TotalValue,AvgPerYear,TotGuarantee,AvgGuarPerYear,PercentGuaranteed):
            scraped_info = {
                'Player': item[0],
                'Position' : item[1],
                'Team' : item[2],
                'TotalValue' : item[3],
                'AvgPerYear' : item[4],
                'TotGuarantee' : item[5],
                'AvgGuarPerYear' : item[6],
                'PercentGuaranteed' : item[7]
                }
    
            
            yield scraped_info
            
