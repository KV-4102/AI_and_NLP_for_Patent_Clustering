import re
from fuzzywuzzy import fuzz
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


stopwords = ["S.A.", "INC.", "LLC", "GMBH", "AB", "OY", "AG", "B.V.", "CO.", "CORP.", "AKTIENGESELLSCHAFT", "L.P.", "A/S", "I.L.P.", "APS", "K.K.", "PTY", "GMBH& CO. KG", "US LP", "CO. LTD.", "UK LIMITED", "LTDA.", "S.P.A.", "N.V.", "PTE. LTD.", "CO. KGAA", "INCORPORATED", 
"ACADEMY FOUNDATION", "CO", "CORPORATION", "AB LIMITED", "PTY LTD", "CO. KG", "LTD.", "A.S.", "A.S.A.", "SA", "LP", "S.A.R.L.", "S.A.S.", "GP", "KG", "GMBH & CO.", "ASA", "S.L.", "S.R.L.", "PLC", "SUISSE", "P.C.", "I L.P.", "LTDA", "A.R.", "NV", "G.M.B.H.", "L.L.C.", "E.V.", 
"NAAMLOZE", "SENAK", "S.A. DE C.V.", "PTE.", "S.A.B.", "S.A.P.I. DE C.V.", "GMBH & CO. KG", "A.G.", "I LP", "PTE LTD", "LLP", "D.B.A.", "LIMITED", "FOUNDATION", "SPA", "R.L.", "KK", "E.V.", "BV", "SCS", "BA", "SRL", "SAIC", "B.V.I.", "I L.L.C.", "INCORPORATED", "KABUSHIKI KAISHA", 
"OHG", "ABSIZE", "KABUSHIKI", "ULC", "S.R.I.", "SOCIETE", "USA", "S.A.R.L", "L.L.C", "SP.", "Z O.O.", "& CO.", "P.A.", "ESQ.", "A.T.A.", "A/S.", "AG.", "AG & CO.","ASSOCIATES","INDUSTRIAL","INDUSTRYACADEMIC", "AG & CO. KGAA", "PTY. LTD.", "GGMBH", "A CALIFORNIA CORPORATION", "AB.", 
"O/Y", "SAS", "FOUNDATION", "ARS HOLDING N.V.", "HOLDING INC.", "AG & CO. KG ÃƒÅ¸", "ET CIE", "NAAMLOZE VENNOOTSCHAP", "LIMITED PARTNERSHIP", "PVT LTD", "PRIVATE LIMITED", "CO. LTD. A CORPORATION", "HOLDING CORPORATION", "HOLDING CO", "GROUP LLC", "GROUP AS", "COMPANY LIMITED", "ASSOCIATES INC.", 
"NP", "TECHNICAL SERVICE LIMITED", "KABUSHISIKI KAISHA", "AG & CO. OHG", "A.S", "AG & CO", "AGENCY", "B.V", "COLLEGE", "COMPANY", "GENTIL", "GROUP", "IND", "INDUSTRY", "INSTITUTE", "INTERNATIONAL", "INVEST", "PARTNERSHIP", "RESEARCH CENTER", "SE", "SOLUTIONS", "SYSTEM", "TECHNOLOGIES", 
"TECHNOLOGY", "COOPERATION", "FOUNDATION", "INSTITUTE", "SCIENCE", "AKTIENGESELLSCHAFT", "AG", "INC", "CO.", "PHARMACEUTICALS", "DELAWARE", "AGENCY", "RESEARCH", "INTERNATIONAL", "CORPORATION", "INDUSTRYACADEMIC", "ACADEMIC", "INDUSTRY", "AB", "LABORATORIES", "AKTIENGELELLSCHAFT", "AS", "LTD", 
"PTE LTD", "GLOBAL LTD", "GLOBAL SERVICES", "COLTD", "HOLDINGS", "DIETSHCLANDGMBH", "DIETSHCLAND GMBH", "SAS", "KOGYO KAISHA", "INDUSTRIES", "KEIKINZOKU KAISHA", "KAKO KAISHA", "SEIKI KAISHA", "SEIKI KABUSHISIKI KAISHA", "PVT", "AKTIEBOLAGET", "HOLDING", "GESELLSCHAFT MBH", "ENTERPRISE", "SERVICES", "INDUSTRIELLES", 
"PVT", "COROPRATION", "AKTIEBOLAG", "AKIENGESELLSCHAFT", "LLC","SEIYAKU KAISHA", "AKTIENGESELLSCHAFT", "KGAA", "COMPANY","ZHANGZHOU","ADVANCED","AND"]

#Removes stopwords from all the lines in data
def remove_stopwords(text, stopwords):
    text = text.upper()
    words = text.split()
    filtered_words = [word for word in words if word.upper() not in stopwords]
    return ' '.join(filtered_words)

#split the input data into patent and assignee and cleans the lines and splits multiple assingees 
def clean_text(text):
    try:
        split_line = text.strip().split("\t")
        patent_number = split_line[0]
        assignee_name=split_line[1]
        splited_assingee = assignee_name.split(";")
        clean_data_list = []
        for x in splited_assingee:
          clean_assignee = remove_stopwords(x.replace("\n","").strip(),stopwords)
          clean_assignee = re.sub(r'[^a-zA-Z0-9\s]', '', clean_assignee)
          clean_assignee = re.sub(r'\s+', ' ', clean_assignee).strip()
          clean_data_list.append([patent_number.strip(),clean_assignee])
      
        return clean_data_list
    except e:
        print(f"Error splitting line: {e}")
        raise Exception("Error splitting line:")



"""Custom similarity function for phase 1 clustering. we find the similarity between 2 strings w.r.t the tokens not in the sorted intersection of both the strings. 
The fuzzy similarity score between the non-intersection parts of the string is calculated"""
def custom_similarity(str1, str2):
  tokens1 = set((str1.split()))  
  tokens2 = set((str2.split()))  
  
  
  # Calculate sorted intersection
  sorted_intersection = sorted(tokens1.intersection(tokens2))
  
  # Remove sorted intersection from each string
  remaining1 = ' '.join(sorted(tokens1 - set(sorted_intersection)))
  remaining2 = ' '.join(sorted(tokens2 - set(sorted_intersection)))
  
  # Calculate similarity score based on remaining tokens
  similarity_score = fuzz.token_set_ratio(remaining1, remaining2)
  return similarity_score

#First stage clustering. Clustering is done based on fuzzy ratio score received from the previous function.
def sort_similar_lines_with_input(data, matcher):
    grouped_lines = {}
    duplicate_assignees = {}  # Dictionary to store duplicate assignees
    for tuple in data:
        original_tuple = tuple  # Store the original input line
        original_line=original_tuple[1]
        patent_number, cleaned_line = tuple # Clean and preprocess the line
        if patent_number is None or cleaned_line is None:
            continue  # Skip processing if cleaning failed
        # Check if assignee already exists in grouped_lines
        if cleaned_line in grouped_lines:
            # Append the line to existing group
            grouped_lines[cleaned_line].append((original_line, patent_number))
        else:
            # Check if assignee is a duplicate
            if cleaned_line in duplicate_assignees:
                # Retrieve existing cluster key for duplicate assignee
                cluster_key = duplicate_assignees[cleaned_line]
                # Append the line to existing cluster
                grouped_lines[cluster_key].append((original_line, patent_number))
            else:
                # Iterate through existing clusters to find a match
                matched = False
                for key in grouped_lines.keys():
                    # Matcher function calls with the appropriate fuzzy algorithm
                    if matcher(cleaned_line, key) > 95:
                        grouped_lines[key].append((original_line, patent_number))  
                        matched = True
                        
                if not matched:
                    grouped_lines[cleaned_line] = [(original_line, patent_number)]  
                    # Add assignee to duplicate_assignees dictionary
                    duplicate_assignees[cleaned_line] = cleaned_line
    return grouped_lines

#used to get singel token companies togther with get_first_word function
def ext_token(keys_custom):
    return [(get_first_word(key[0]) if key and len(key[0])>1 else ("None", *key[1:]), *key[1:]) for key in keys_custom
           if key[0] is not None and get_first_word(key[0])not in stopwords2]


def get_first_word(word):
 if(len(word.split()))==1:
    key=word
    return key


#To tag single token words with nltk pos tagger
def tag_keys(first_word_list):
    tagged_keys = []
    for key in first_word_list:
        if key[0] is not None:
            try:
                first_word = key[0].lower()  
                tokens = word_tokenize(first_word)
                tagged_first_word = pos_tag(tokens)[0]
                tagged_key = (tagged_first_word, *key[1:])
                tagged_keys.append(tagged_key)
            except AttributeError:  
                pass  
    return tagged_keys


stopwords2=['HONG KONG', 'MACAU', 'BEIJING', 'CHONGQING', 'SHANGHAI', 'TIANJIN', 'ANQING', 'BENGBU', 'BOZHOU', 'CHAOHU', 'CHIZHOU', 'CHUZHOU', 'FUYANG', 'HEFEI', 'HUAIBEI', 'HUAINAN', 'HUANGSHAN', 'JIESHOU', "LU'AN", "MA'ANSHAN", 'MINGGUANG', 'NINGGUO', 'QIANSHAN', 'SUZHOU', 'TIANCHANG', 'TONGCHENG', 'TONGLING', 'WUHU', 'XUANCHENG', "FU'AN", 'FUDING', 'FUQING', 'FUZHOU', "JIAN'OU", 'JINJIANG', 'LONGHAI', 'LONGYAN', "NAN'AN", 'NANPING', 'NINGDE', 'PUTIAN', 'QUANZHOU', 'SANMING', 'SHAOWU', 'SHISHI', 'WUYISHAN', 'XIAMEN', "YONG'AN", 'ZHANGPING', 'ZHANGZHOU', 'BAIYIN', 'DINGXI', 'DUNHUANG', 'HEZUO', 'HUATING', 'JIAYUGUAN', 'JINCHANG', 'JIUQUAN', 'LANZHOU', 'LINXIA', 'LONGNAN', 'PINGLIANG', 'QINGYANG', 'TIANSHUI', 'WUWEI', 'YUMEN', 'ZHANGYE', 'CHAOZHOU', 'DONGGUAN', 'ENPING', 'FOSHAN', 'GAOZHOU', 'GUANGZHOU', 'HESHAN', 'HEYUAN', 'HUAZHOU', 'HUIZHOU', 'JIANGMEN', 'JIEYANG', 'KAIPING', 'LECHANG', 'LEIZHOU', 'LIANJIANG', 'LIANZHOU', 'LUFENG', 'LUODING', 'MAOMING', 'MEIZHOU', 'NANXIONG', 'PUNING', 'QINGYUAN', 'SHANTOU', 'SHANWEI', 'SHAOGUAN', 'SHENZHEN', 'SIHUI', 'TAISHAN', 'WUCHUAN', 'XINGNING', 'XINYI', 'YANGCHUN', 'YANGJIANG', 'YINGDE', 'YUNFU', 'ZHANJIANG', 'ZHAOQING', 'ZHONGSHAN', 'ZHUHAI', 'BAISE', 'BEIHAI', 'BEILIU', 'CENXI', 'CHONGZUO', 'DONGXING', 'FANGCHENGGANG', 'GUIGANG', 'GUILIN', 'GUIPING', 'HECHI', 'HESHAN', 'HEZHOU', 'JINGXI', 'LAIBIN', 'LIPU', 'LIUZHOU', 'NANNING', 'PINGXIANG', 'QINZHOU', 'WUZHOU', 'YULIN', 'ANSHUN', 'BIJIE', 'CHISHUI', 'DUYUN', 'FUQUAN', 'GUIYANG', 'KAILI', 'LIUPANSHUI', 'PANZHOU', 'QINGZHEN', 'RENHUAI', 'TONGREN', 'XINGREN', 'XINGYI', 'ZUNYI', 'DANZHOU', 'DONGFANG', 'HAIKOU', 'QIONGHAI', 'SANSHA', 'SANYA', 'WANNING', 'WENCHANG', 'WUZHISHAN', 'ANGUO', 'BAODING', 'BAZHOU', 'BOTOU', 'CANGZHOU', 'CHENGDE', 'DINGZHOU', 'GAOBEIDIAN', 'HANDAN', 'HENGSHUI', 'HEJIAN', 'HUANGHUA', 'JINZHOU', 'LANGFANG', 'LUANZHOU', 'NANGONG', 'PINGQUAN', "QIAN'AN", 'QINHUANGDAO', 'RENQIU', 'SANHE', 'SHAHE', 'SHENZHOU', 'SHIJIAZHUANG', 'TANGSHAN', 'XINJI', "WU'AN", 'XINGTAI', 'XINLE', 'ZHANGJIAKOU', 'ZHUOZHOU', 'ZUNHUA', 'ANDA', "BEI'AN", 'DAQING', 'DONGNING', 'FUJIN', 'FUYUAN', 'HAILIN', 'HAILUN', 'HARBIN', 'HEGANG', 'HEIHE', 'HULIN', 'JIAMUSI', 'JIXI', 'MISHAN', 'MOHE', 'MUDANJIANG', 'MULING', 'NEHE', "NING'AN", 'QIQIHAR', 'QITAIHE', 'SHANGZHI', 'SHUANGYASHAN', 'SUIFENHE', 'SUIHUA', 'TIELI', 'TONGJIANG', 'WUCHANG', 'WUDALIANCHI', 'YICHUN', 'ZHAODONG', 'ANYANG', 'CHANGGE', 'DENGFENG', 'DENGZHOU', 'GONGYI', 'HEBI', 'HUIXIAN', 'JIAOZUO', 'JIYUAN', 'KAIFENG', 'LINGBAO', 'LINZHOU', 'LUOHE', 'LUOYANG', 'MENGZHOU', 'NANYANG', 'PINGDINGSHAN', 'PUYANG', 'QINYANG', 'RUZHOU', 'SANMENXIA', 'SHANGQIU', 'WEIHUI', 'WUGANG', 'XIANGCHENG', 'XINGYANG', 'XINMI', 'XINXIANG', 'XINYANG', 'XINZHENG', 'XUCHANG', 'YANSHI', 'YIMA', 'YONGCHENG', 'YUZHOU', 'ZHENGZHOU', 'ZHOUKOU', 'ZHUMADIAN', 'ANLU', 'CHIBI', 'DANGYANG', 'DANJIANGKOU', 'DAYE', 'ENSHI', 'EZHOU', 'GUANGSHUI', 'HANCHUAN', 'HONGHU', 'HUANGGANG', 'HUANGSHI', 'JINGMEN', 'JINGSHAN', 'JINGZHOU', 'LAOHEKOU', 'LICHUAN', 'MACHENG', 'QIANJIANG', 'SHISHOU', 'SHIYAN', 'SUIZHOU', 'SONGZI', 'TIANMEN', 'WUHAN', 'WUXUE', 'XIANGYANG', 'XIANNING', 'XIANTAO', 'XIAOGAN', 'YICHANG', 'YICHENG', 'YIDU', 'YINGCHENG', 'ZAOYANG', 'ZHIJIANG', 'ZHONGXIANG', 'CHANGDE', 'CHANGNING', 'CHANGSHA', 'CHENZHOU', 'HENGYANG', 'HONGJIANG', 'HUAIHUA', 'JINSHI', 'JISHOU', 'LEIYANG', 'LENGSHUIJIANG', 'LIANYUAN', 'LILING', 'LINXIANG', 'LIUYANG', 'LOUDI', 'MILUO', 'NINGXIANG', 'SHAOSHAN', 'SHAOYANG', 'WUGANG', 'XIANGTAN', 'XIANGXIANG', 'YIYANG', 'YONGZHOU', 'YUANJIANG', 'YUEYANG', 'ZHANGJIAJIE', 'ZHUZHOU', 'ZIXING', 'ARXAN', 'BAOTOU', 'BAYANNUR', 'CHIFENG', 'ERENHOT', 'ERGUN', 'FENGZHEN', 'GENHE', 'HOHHOT', 'HOLINGOL', 'HULUNBUIR', 'MANZHOULI', 'ORDOS', 'TONGLIAO', 'ULANHOT', 'ULANQAB', 'WUHAI', 'XILINHOT', 'YAKESHI', 'ZHALANTUN', 'CHANGSHU', 'CHANGZHOU', 'DANYANG', 'DONGTAI', 'GAOYOU', "HAI'AN", 'HAIMEN', "HUAI'AN", 'JIANGYIN', 'JINGJIANG', 'JURONG', 'LIYANG', 'LIANYUNGANG', 'KUNSHAN', 'NANJING', 'NANTONG', 'PIZHOU', 'QIDONG', 'RUGAO', 'SUQIAN', 'SUZHOU', 'TAICANG', 'TAIXING', 'TAIZHOU', 'WUXI', 'XINGHUA', 'XINYI', 'XUZHOU', 'YANCHENG', 'YANGZHONG', 'YANGZHOU', 'YIXING', 'YIZHENG', 'ZHANGJIAGANG', 'ZHENJIANG', 'DEXING', 'FENGCHENG', 'FUZHOU', 'GANZHOU', "GAO'AN", 'GONGQINGCHENG', 'GUIXI', "JI'AN", 'JINGDEZHEN', 'JINGGANGSHAN', 'JIUJIANG', 'LEPING', 'LUSHAN', 'NANCHANG', 'PINGXIANG', 'RUICHANG', 'RUIJIN', 'SHANGRAO', 'XINYU', 'YICHUN', 'YINGTAN', 'ZHANGSHU', 'BAICHENG', 'BAISHAN', 'CHANGCHUN', "DA'AN", 'DEHUI', 'DUNHUA', 'FUYU', 'GONGZHULING', 'HELONG', 'HUADIAN', 'HUNCHUN', "JI'AN", 'JIAOHE', 'JILIN', 'LIAOYUAN', 'LINJIANG', 'LONGJING', 'MEIHEKOU', 'PANSHI', 'SHUANGLIAO', 'SHULAN', 'SIPING', 'SONGYUAN', 'TAONAN', 'TONGHUA', 'TUMEN', 'YANJI', 'YUSHU', 'ANSHAN', 'BENXI', 'BEIPIAO', 'BEIZHEN', 'CHAOYANG', 'DALIAN', 'DANDONG', 'DASHIQIAO', 'DENGTA', 'DIAOBINGSHAN', 'DONGGANG', 'FENGCHENG', 'FUSHUN', 'FUXIN', 'GAIZHOU', 'HAICHENG', 'HULUDAO', 'JINZHOU', 'KAIYUAN', 'LIAOYANG', 'LINGHAI', 'LINGYUAN', 'PANJIN', 'SHENYANG', 'TIELING', 'WAFANGDIAN', 'XINGCHENG', 'XINMIN', 'YINGKOU', 'ZHUANGHE', 'GUYUAN', 'LINGWU', 'QINGTONGXIA', 'SHIZUISHAN', 'WUZHONG', 'YINCHUAN', 'ZHONGWEI', 'DELINGHA', 'GOLMUD', 'HAIDONG', 'MANGNAI', 'XINING', 'YUSHU', 'ANKANG', 'BAOJI', 'BINZHOU', 'HANCHENG', 'HANZHONG', 'HUAYIN', 'SHANGLUO', 'SHENMU', 'TONGCHUAN', 'WEINAN', "XI'AN", 'XIANYANG', 'XINGPING', "YAN'AN", 'YULIN', 'ANQIU', 'BINZHOU', 'CHANGYI', 'DEZHOU', 'DONGYING', 'FEICHENG', 'GAOMI', 'HAIYANG', 'HEZE', 'JIAOZHOU', 'JINAN', 'JINING', 'LAIXI', 'LAIYANG', 'LAIZHOU', 'LELING', 'LIAOCHENG', 'LINQING', 'LINYI', 'LONGKOU', 'PENGLAI', 'PINGDU', 'QINGDAO', 'QINGZHOU', 'QIXIA', 'QUFU', 'RIZHAO', 'RONGCHENG', 'RUSHAN', 'SHOUGUANG', "TAI'AN", 'TENGZHOU', 'WEIFANG', 'WEIHAI', 'XINTAI', 'YANTAI', 'YUCHENG', 'ZAOZHUANG', 'ZHAOYUAN', 'ZHUCHENG', 'ZIBO', 'ZOUCHENG', 'ZOUPING', 'CHANGZHI', 'DATONG', 'FENYANG', 'GAOPING', 'GUJIAO', 'HEJIN', 'HOUMA', 'HUAIREN', 'HUOZHOU', 'JIEXIU', 'JINCHENG', 'JINZHONG', 'LINFEN', 'LÜLIANG', 'SHUOZHOU', 'TAIYUAN', 'XIAOYI', 'XINZHOU', 'YANGQUAN', 'YONGJI', 'YUNCHENG', 'YUANPING', 'BARKAM', 'BAZHONG', 'CHENGDU', 'CHONGZHOU', 'DAZHOU', 'DEYANG', 'DUJIANGYAN', 'EMEISHAN', "GUANG'AN", 'GUANGHAN', 'GUANGYUAN', 'HUAYING', 'JIANGYOU', 'JIANYANG', 'KANGDING', 'LANGZHONG', 'LESHAN', 'LONGCHANG', 'LUZHOU', 'MIANZHU', 'MEISHAN', 'MIANYANG', 'NANCHONG', 'NEIJIANG', 'PANZHIHUA', 'PENGZHOU', 'QIONGLAI', 'SHIFANG', 'SUINING', 'WANYUAN', 'XICHANG', "YA'AN", 'YIBIN', 'ZIGONG', 'ZIYANG', 'LHASA', 'NAGQU', 'NYINGCHI', 'QAMDO', 'SHANNAN', 'XIGAZÊ', 'AKSU', 'ALASHANKOU', 'ALTAY', 'ARAL', 'ARTUX', 'BEITUN', 'BOLE', 'CHANGJI', 'FUKANG', 'HAMI', 'HOTAN', 'KARAMAY', 'KASHGAR', 'KHORGAS', 'KOKDALA', 'KORLA', 'KUYTUN', 'KUNYU', 'SHIHEZI', 'SHUANGHE', 'TACHENG', 'TIEMENGUAN', 'TUMXUK', 'TURPAN', 'ÜRÜMQI', 'WUJIAQU', 'WUSU', 'YINING', 'ANNING', 'BAOSHAN', 'CHUXIONG', 'DALI', 'GEJIU', 'JINGHONG', 'KAIYUAN', 'KUNMING', 'LINCANG', 'LIJIANG', 'LUSHUI', 'MANGSHI', 'MENGZI', 'MILE', "PU'ER", 'QUJING', 'RUILI', 'SHANGRI-LA', 'SHUIFU', 'TENGCHONG', 'WENSHAN', 'XUANWEI', 'YUXI', 'ZHAOTONG', 'CIXI', 'DONGYANG', 'HAINING', 'HANGZHOU', 'HUZHOU', 'JIANDE', 'JIANGSHAN', 'JIAXING', 'JINHUA', 'LANXI', 'LINHAI', 'LISHUI', 'LONGQUAN', 'NINGBO', 'PINGHU', 'QUZHOU', 'RUIAN', 'SHAOXING', 'SHENGZHOU', 'TAIZHOU', 'TONGXIANG', 'WENLING', 'WENZHOU', 'YIWU', 'YONGKANG', 'YUEQING', 'YUHUAN', 'YUYAO', 'ZHOUSHAN', 'ZHUJI', 'ACADEMIA', 'ACADEMISH', 'ACCESS', 'ENGINEERING', 'CORPS', 'ACADEMY', 'ACTION', 'ADAM', 'ADM', 'ADVANCE', 'ADVANCED', 'ADVANCEMENT', 'ADVANTAGE', 'AERO', 'AGC', 'AGILE', 'AGRI', 'AGRICULTURE', 'AGRO', 'AIR', 'AL', 'ALERT', 'ALEXANDER', 'ALFA', 'ALLEN', 'ALLIANCE', 'ALLIANT', 'AMBIENT', 'ANALOGIC', 'ANALYSIS', 'ANDREW', 'ANGEL', 'ANN', 'AO', 'APEX', 'APOLLO', 'ARBOR', 'ARROWHEAD', 'ART', 'ASPECT', 'AUTO', 'AV', 'AXIS', 'AZ', 'BAKER', 'BANK', 'BASF', 'BEAM', 'BOARD', 'DANA', 'DEUTSCHE', 'DIGITAL', 'DONG', 'DR', 'IMAGE', 'JOHN', 'MEMBRANE', 'NETWORK', 'OLYMPUS', 'PACIFIC', 'REPUBLIC', 'ROCHE', 'ROBERT', 'ROLL', 'ROLLER', 'ROYAL', 'SPECIALTY', 'SPACE', 'STUDIO', 'UNIVERSITAT', 'UNIVERSITY', 'VALEO', 'WILLIAM', 'ZHEJIANGAFGHANISTAN', 'ALBANIA', 'ALGERIA', 'ANDORRA', 'ANGOLA', 'ANTIGUA AND BARBUDA', 'ARGENTINA', 'ARMENIA', 'AUSTRALIA', 'AUSTRIA', 'AZERBAIJAN', 'BAHAMAS', 'BAHRAIN', 'BANGLADESH', 'BARBADOS', 'BELARUS', 'BELGIUM', 'BELIZE', 'BENIN', 'BHUTAN', 'BOLIVIA', 'BOSNIA AND HERZEGOVINA', 'BOTSWANA', 'BRAZIL', 'BRUNEI', 'BULGARIA', 'BURKINA FASO', 'BURUNDI', 'CABO VERDE', 'CAMBODIA', 'CAMEROON', 'CANADA', 'CENTRAL AFRICAN REPUBLIC', 'CHAD', 'CHILE', 'CHINA', 'COLOMBIA', 'COMOROS', 'CONGO, DEMOCRATIC REPUBLIC OF THE', 'CONGO, REPUBLIC OF THE', 'COSTA RICA', 'CROATIA', 'CUBA', 'CYPRUS', 'CZECH REPUBLIC', 'DENMARK', 'DJIBOUTI', 'DOMINICA', 'DOMINICAN REPUBLIC', 'EAST TIMOR', 'ECUADOR', 'EGYPT', 'EL SALVADOR', 'EQUATORIAL GUINEA', 'ERITREA', 'ESTONIA', 'ESWATINI', 'ETHIOPIA', 'FIJI', 'FINLAND', 'FRANCE', 'GABON', 'GAMBIA', 'GEORGIA', 'GERMANY', 'GHANA', 'GREECE', 'GRENADA', 'GUATEMALA', 'GUINEA', 'GUINEA-BISSAU', 'GUYANA', 'HAITI', 'HONDURAS', 'HUNGARY', 'ICELAND', 'INDIA', 'INDONESIA', 'IRAN', 'IRAQ', 'IRELAND', 'ISRAEL', 'ITALY', "IVORY COAST (CÔTE D'IVOIRE)", 'JAMAICA', 'JAPAN', 'JORDAN', 'KAZAKHSTAN', 'KENYA', 'KIRIBATI', 'KOREA, NORTH', 'KOREA, SOUTH', 'KOSOVO', 'KUWAIT', 'KYRGYZSTAN', 'LAOS', 'LATVIA', 'LEBANON', 'LESOTHO', 'LIBERIA', 'LIBYA', 'LIECHTENSTEIN', 'LITHUANIA', 'LUXEMBOURG', 'MADAGASCAR', 'MALAWI', 'MALAYSIA', 'MALDIVES', 'MALI', 'MALTA', 'MARSHALL ISLANDS', 'MAURITANIA', 'MAURITIUS', 'MEXICO', 'MICRONESIA', 'MOLDOVA', 'MONACO', 'MONGOLIA', 'MONTENEGRO', 'MOROCCO', 'MOZAMBIQUE', 'MYANMAR', 'NAMIBIA', 'NAURU', 'NEPAL', 'NETHERLANDS', 'NEW ZEALAND', 'NICARAGUA', 'NIGER', 'NIGERIA', 'NORTH MACEDONIA', 'NORWAY', 'OMAN', 'PAKISTAN', 'PALAU', 'PANAMA', 'PAPUA NEW GUINEA', 'PARAGUAY', 'PERU', 'PHILIPPINES', 'POLAND', 'PORTUGAL', 'QATAR', 'ROMANIA', 'RUSSIA', 'RWANDA', 'SAINT KITTS AND NEVIS', 'SAINT LUCIA', 'SAINT VINCENT AND THE GRENADINES', 'SAMOA', 'SAN MARINO', 'SAO TOME AND PRINCIPE', 'SAUDI ARABIA', 'SENEGAL', 'SERBIA', 'SEYCHELLES', 'SIERRA LEONE', 'SINGAPORE', 'SLOVAKIA', 'SLOVENIA', 'TOKYO', 'SOLOMON ISLANDS', 'SOMALIA', 'SOUTH AFRICA', 'SOUTH SUDAN', 'SPAIN', 'SRI LANKA', 'SUDAN', 'SURINAME', 'SWEDEN', 'SWITZERLAND', 'SYRIA', 'TAIWAN', 'TAJIKISTAN', 'TANZANIA', 'THAILAND', 'TOGO', 'TONGA', 'TRINIDAD AND TOBAGO', 'TUNISIA', 'TURKEY', 'TURKMENISTAN', 'TUVALU', 'UGANDA', 'UKRAINE', 'UNITED ARAB EMIRATES', 'UNITED KINGDOM', 'UNITED STATES', 'URUGUAY', 'UZBEKISTAN', 'VANUATU', 'VATICAN CITY', 'VENEZUELA', 'VIETNAM', 'YEMEN', 'ZAMBIA', 'ZIMBABWE', 'ABSOLUTE', 'ADEKA', 'AI', 'ZF', 'AMERICA','ACCESSED','ENGINEERING','AND']


#clustering is done again w.r.t to keys(clusters) received from the first clustering phase. Pos tagger is used so that only strings tagged as NNP is considered as cluster name while clustering
def generate_second_clusters(keys_custom, tagged_words):

  second_grouped_lines = {}

  # Cluster words tagged as NNP or NN
  for tagged_word in tagged_words:
    if tagged_word[0][1] == "NNP" or tagged_word[0][1] == "NN":
      # Extract the word as the key
      word = tagged_word[0][0]
      # Append the entire tagged word
      second_grouped_lines.setdefault(word, []).append(word)

  threshold = 100

  # Loop through custom keys (including non-NNP/NN)
  for key_tuple in keys_custom:
    key_to_find = key_tuple[0]
    matched = False

    # Find a matching cluster using fuzzy ratio
    for cluster_key in second_grouped_lines.keys():
      similarity_score = fuzz.token_set_ratio(key_to_find, cluster_key)
      #if len(key_to_find.split())<4:
      if similarity_score == threshold:
          second_grouped_lines[cluster_key].append(key_tuple)
          matched = True 
    # If no match is found, add the entire key_tuple
    if not matched:
      second_grouped_lines[key_to_find] = [key_tuple]

  return second_grouped_lines



if __name__ =='__main__':

    with open("/home/krishna/Desktop/gl_std_assignee/sorted_assignees.txt", "r", encoding="utf-8") as file:
        data = file.readlines()
        data_list = []
        for line in data:
            try:
                multiple_patent_assingee = clean_text(line)
                data_list.extend(multiple_patent_assingee)

            except Exception as e:
                continue

        grouped_lines_custom = sort_similar_lines_with_input(data_list, custom_similarity)
        keys_custom = list(grouped_lines_custom.items())
        keys_custom.sort()
        first_word_list = ext_token(keys_custom)
        tagged_words= tag_keys(first_word_list)
        singleton_cluster_name = "Not clustered in phase 2:" 

        second_grouped_lines_custom = generate_second_clusters(keys_custom,tagged_words)
        with open("norm_op.txt", "w") as file:
          for value in second_grouped_lines_custom.values():
            file.write(str(value) + "\n")
          
        print("Clustering successful")

